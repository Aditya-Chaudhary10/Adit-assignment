"""Domain models for the outbound healthcare voice agent.

Kept free of LiveKit/Opik imports so it can be used by the dispatcher, the
agent worker and the analysis layer alike.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PATIENTS_FILE = os.path.join(DATA_DIR, "patients.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Biomarker:
    key: str
    label: str
    value: Any
    unit: str
    reference_range: str
    status: str  # normal | borderline | high | low
    plain_language: str = ""

    @property
    def is_abnormal(self) -> bool:
        return self.status.lower() not in ("normal", "ok", "in_range")

    def spoken(self) -> str:
        """How the value should be read out loud on a phone call."""
        return f"{self.label} is {self.value} {self.unit}".strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatientRecord:
    patient_id: str
    name: str
    phone_number: str
    biomarkers: list[Biomarker] = field(default_factory=list)
    age: int | None = None
    preferred_language: str = "English"
    care_program: str = ""
    last_lab_date: str = ""
    recommended_specialty: str = "Internal Medicine"
    urgency: str = "routine"
    consent_on_file: bool = True

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else "there"

    @property
    def abnormal_biomarkers(self) -> list[Biomarker]:
        return [b for b in self.biomarkers if b.is_abnormal]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PatientRecord":
        data = dict(raw)
        data["biomarkers"] = [Biomarker(**b) for b in raw.get("biomarkers", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out

    def biomarker_briefing(self) -> str:
        """A compact, unambiguous block the LLM reads verbatim values from."""
        if not self.biomarkers:
            return "No biomarker results on file."
        lines = []
        for b in self.biomarkers:
            lines.append(
                f"- {b.label}: {b.value} {b.unit} "
                f"(normal range {b.reference_range}) -> {b.status.upper()}. "
                f"Explain it as: {b.plain_language}"
            )
        return "\n".join(lines)


@dataclass
class CallContext:
    """Everything the worker needs to place and describe one outbound call."""

    patient: PatientRecord
    call_id: str
    clinic_name: str = "Northside Metabolic Health"
    agent_display_name: str = "Riley"
    transfer_to: str | None = None
    simulate: bool = False           # skip SIP, run in-room only (browser/console demo)
    record_call: bool = False
    started_at: str = field(default_factory=utc_now_iso)

    def to_metadata(self) -> str:
        """Serialized onto the LiveKit job so the worker can rebuild it."""
        return json.dumps(
            {
                "call_id": self.call_id,
                "clinic_name": self.clinic_name,
                "agent_display_name": self.agent_display_name,
                "transfer_to": self.transfer_to,
                "simulate": self.simulate,
                "record_call": self.record_call,
                "started_at": self.started_at,
                "patient": self.patient.to_dict(),
            }
        )

    @classmethod
    def from_metadata(cls, metadata: str) -> "CallContext":
        raw = json.loads(metadata)
        patient = PatientRecord.from_dict(raw["patient"])
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in raw.items() if k in known and k != "patient"}
        return cls(patient=patient, **kwargs)


def load_patients(path: str = PATIENTS_FILE) -> list[PatientRecord]:
    with open(path, "r", encoding="utf-8") as fh:
        return [PatientRecord.from_dict(p) for p in json.load(fh)]


def find_patient(identifier: str, path: str = PATIENTS_FILE) -> PatientRecord:
    """Look a patient up by id, exact phone number, or case-insensitive name."""
    patients = load_patients(path)
    ident = identifier.strip().lower()
    for p in patients:
        if ident in (p.patient_id.lower(), p.phone_number.lower(), p.name.lower()):
            return p
    for p in patients:
        if ident in p.name.lower():
            return p
    raise KeyError(
        f"No patient matched {identifier!r}. "
        f"Known: {', '.join(f'{p.patient_id}/{p.name}' for p in patients)}"
    )


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

_PLACEHOLDER_MARKERS = ("xxxx", "your-", "your_", "<", "changeme")


def _is_placeholder(value: str) -> bool:
    """True for a value that is still the .env.example template text.

    A half-filled .env is the most common setup mistake, and leaving the
    template string in place produces a confusing 401 from the provider rather
    than an obvious 'not configured'. Treating these as unset means the code
    takes its documented no-key path instead.
    """
    low = value.strip().lower()
    if not low:
        return True
    return any(marker in low for marker in _PLACEHOLDER_MARKERS)


def load_environment(*paths: str) -> list[str]:
    """Load .env files, then drop any variable still holding template text.

    Returns the names of the variables that were ignored, so callers can warn.
    """
    from dotenv import load_dotenv

    for path in paths or (".env.local", ".env"):
        load_dotenv(path, override=False)

    watched = (
        "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
        "GROQ_API_KEY", "OPENAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY",
        "OPIK_API_KEY", "OPIK_WORKSPACE", "SIP_OUTBOUND_TRUNK_ID",
    )
    ignored = []
    for name in watched:
        value = os.environ.get(name)
        if value is None:
            continue
        if _is_placeholder(value):
            del os.environ[name]
            if value.strip():
                ignored.append(name)
            continue
        # A key pasted into .env often carries a trailing space or newline, which
        # the provider rejects as an invalid credential with no useful message.
        stripped = value.strip().strip('"').strip("'")
        if stripped != value:
            os.environ[name] = stripped
    return ignored


def _silence_windows_asyncio_shutdown_noise() -> None:
    """Stop Windows printing a fake traceback after every clean exit.

    On Windows, asyncio's proactor transports are finalized after the event loop
    has already closed, and their __del__ raises "Event loop is closed". The work
    is already done and nothing is actually wrong, but it prints a red traceback
    that makes a successful run look like a crash. Harmless everywhere else.
    """
    import sys

    if not sys.platform.startswith("win"):
        return
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
    except ImportError:
        return

    original = _ProactorBasePipeTransport.__del__

    def quiet_del(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            original(self, *args, **kwargs)
        except RuntimeError as exc:
            if "Event loop is closed" not in str(exc):
                raise

    _ProactorBasePipeTransport.__del__ = quiet_del  # type: ignore[assignment]


def configure_console_encoding() -> None:
    """Force UTF-8 on stdout/stderr.

    Windows consoles default to a legacy code page such as cp1252. LLM output
    routinely contains characters outside it, for example the narrow no-break
    space U+202F, and printing one raises UnicodeEncodeError and kills the run.
    Replacing unencodable characters is always better than crashing on a log line.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    _silence_windows_asyncio_shutdown_noise()
