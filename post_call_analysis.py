"""Post-call analysis: turn a finished conversation into a structured outcome.

The design is deliberately two-layered:

1. A deterministic pass over the tool-call record. Whether an appointment was
   booked is a *fact* the agent recorded by calling `book_appointment`, so we
   read it from the tool results rather than asking an LLM to infer it. This is
   the authoritative answer and it cannot hallucinate.
2. An LLM pass over the transcript for the parts that genuinely need judgment:
   sentiment, objections, whether the biomarkers were actually communicated,
   a summary, and the recommended follow-up.

If no LLM key is configured the module still returns a valid analysis using
layer 1 only, so the pipeline never hard-fails on the analysis step.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from typing import Any

logger = logging.getLogger("post-call-analysis")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def resolve_analysis_provider() -> tuple[str, str, str, str] | None:
    """Pick the LLM for the analysis pass.

    Returns (provider, api_key, base_url, model), or None when no key is set.
    Groq and OpenAI both speak the OpenAI chat-completions API, so one client
    covers both. The models chosen are the ones that support strict
    `json_schema` structured output, which is what makes this pass reliable.
    """
    explicit = os.getenv("ANALYSIS_MODEL", "")

    if os.getenv("GROQ_API_KEY"):
        return (
            "groq",
            os.environ["GROQ_API_KEY"],
            os.getenv("GROQ_BASE_URL", GROQ_BASE_URL),
            explicit or "openai/gpt-oss-120b",
        )
    if os.getenv("OPENAI_API_KEY"):
        return (
            "openai",
            os.environ["OPENAI_API_KEY"],
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            explicit or "gpt-4o-mini",
        )
    return None

OUTCOMES = [
    "appointment_booked",
    "appointment_declined",
    "callback_requested",
    "no_answer",
    "voicemail",
    "wrong_number",
    "call_dropped",
    "patient_hung_up",
    "not_interested",
    "needs_human_followup",
]

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "outcome",
        "appointment_booked",
        "summary",
        "patient_sentiment",
        "biomarkers_communicated",
        "patient_concerns",
        "objections",
        "follow_up_required",
        "recommended_next_action",
        "call_quality_issues",
        "escalation_needed",
        "confidence",
    ],
    "properties": {
        "outcome": {"type": "string", "enum": OUTCOMES},
        "appointment_booked": {"type": "boolean"},
        "summary": {"type": "string", "description": "Three to four sentence factual recap."},
        "patient_sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative", "distressed", "unknown"],
        },
        "biomarkers_communicated": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Biomarker labels the agent actually stated to the patient.",
        },
        "patient_concerns": {"type": "array", "items": {"type": "string"}},
        "objections": {"type": "array", "items": {"type": "string"}},
        "follow_up_required": {"type": "boolean"},
        "recommended_next_action": {"type": "string"},
        "call_quality_issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Agent-side problems only: talked over the patient, repeated itself, "
                "misread a value, ignored a question."
            ),
        },
        "escalation_needed": {
            "type": "boolean",
            "description": "True if the patient described symptoms or distress needing a human clinician.",
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

SYSTEM_PROMPT = """You analyze completed outbound healthcare phone calls made by an AI voice agent.

The agent called a patient to (a) explain recent lab biomarker results and (b) book a doctor consultation.

You will be given the patient's source record, the full transcript, and the exact tool calls the agent executed.

Rules:
- Judge only from the transcript and the tool record. Never invent details.
- appointment_booked is true only if a booking tool call succeeded. The tool record is authoritative.
- biomarkers_communicated lists only biomarkers the agent actually spoke aloud.
- call_quality_issues is about the AGENT's performance, not the patient's.
- escalation_needed is true if the patient reported symptoms, distress, or a question requiring a clinician.
- If the transcript is empty or only the agent spoke, the outcome is no_answer or voicemail.

Return only the structured object."""


def coerce_tool_output(output: Any) -> Any:
    """Turn a tool result back into a dict whatever shape it arrives in.

    LiveKit serializes a tool's return value with `str()`, so a dict comes back
    as a Python repr with single quotes, which `json.loads` rejects. Parsing only
    as JSON silently reduced every booking to an opaque string and made
    `appointment_booked` permanently False even on calls that booked
    successfully. Try JSON first, then Python literals.
    """
    if not isinstance(output, str):
        return output
    text = output.strip()
    if not text.startswith(("{", "[")):
        return output
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return {"raw": output}


def _authoritative_booking(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Read booking truth straight out of the tool record."""
    booking: dict[str, Any] = {"booked": False, "details": None, "attempted": False}
    for call in tool_calls:
        name = (call.get("name") or "").lower()
        if "book" not in name and "schedule" not in name:
            continue
        booking["attempted"] = True
        output = coerce_tool_output(call.get("output"))
        if isinstance(output, dict) and output.get("status") == "confirmed":
            booking["booked"] = True
            booking["details"] = output
    return booking


def _transcript_text(transcript: list[dict[str, Any]]) -> str:
    lines = []
    for turn in transcript:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        if not str(content).strip():
            continue
        speaker = {"assistant": "AGENT", "user": "PATIENT"}.get(role, str(role).upper())
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "(no speech was exchanged)"


def _tool_calls_text(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        return "(the agent executed no tools)"
    out = []
    for c in tool_calls:
        args = json.dumps(c.get("arguments"), default=str)
        result = json.dumps(c.get("output"), default=str)
        out.append(f"- {c.get('name')}(args={args}) -> {result}")
    return "\n".join(out)


def _fallback_analysis(
    booking: dict[str, Any],
    transcript: list[dict[str, Any]],
    telemetry: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    patient_spoke = any(
        t.get("role") == "user" and str(t.get("content", "")).strip() for t in transcript
    )
    if booking["booked"]:
        outcome = "appointment_booked"
    elif not patient_spoke:
        outcome = "no_answer"
    elif booking["attempted"]:
        outcome = "needs_human_followup"
    else:
        outcome = "appointment_declined"
    return {
        "outcome": outcome,
        "appointment_booked": booking["booked"],
        "summary": (
            f"Deterministic analysis only ({reason}). "
            f"Booking tool confirmed: {booking['booked']}."
        ),
        "patient_sentiment": "unknown",
        "biomarkers_communicated": [],
        "patient_concerns": [],
        "objections": [],
        "follow_up_required": not booking["booked"],
        "recommended_next_action": "Manual review: LLM analysis unavailable.",
        "call_quality_issues": [],
        "escalation_needed": False,
        "confidence": 0.3,
        "analysis_mode": "deterministic_fallback",
        "appointment_details": booking["details"],
        "telemetry": telemetry,
    }


async def analyze_call(
    *,
    patient: dict[str, Any],
    transcript: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    telemetry: dict[str, Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Produce the structured post-call analysis object."""
    telemetry = telemetry or {}
    booking = _authoritative_booking(tool_calls)

    provider = resolve_analysis_provider()
    if provider is None:
        return _fallback_analysis(
            booking, transcript, telemetry, "no GROQ_API_KEY or OPENAI_API_KEY set"
        )
    provider_name, api_key, base_url, default_model = provider
    model = model or default_model

    user_prompt = (
        "PATIENT RECORD (source of truth for biomarker values):\n"
        f"{json.dumps(patient, indent=2, default=str)}\n\n"
        "CALL TELEMETRY:\n"
        f"{json.dumps(telemetry, indent=2, default=str)}\n\n"
        "TOOL CALLS EXECUTED BY THE AGENT:\n"
        f"{_tool_calls_text(tool_calls)}\n\n"
        "TRANSCRIPT:\n"
        f"{_transcript_text(transcript)}\n"
    )

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "post_call_analysis",
                    "strict": True,
                    "schema": ANALYSIS_SCHEMA,
                },
            },
        )
        analysis = json.loads(resp.choices[0].message.content)
        analysis["analysis_mode"] = "llm"
        analysis["analysis_model"] = model
        analysis["analysis_provider"] = provider_name
    except Exception as exc:  # noqa: BLE001 - analysis must never crash the pipeline
        logger.exception("LLM post-call analysis failed, falling back to deterministic")
        return _fallback_analysis(booking, transcript, telemetry, f"LLM error: {exc}")

    # The tool record overrides the model on the one fact that must not be guessed.
    if analysis.get("appointment_booked") != booking["booked"]:
        logger.warning(
            "LLM reported appointment_booked=%s but the tool record says %s; trusting tool record",
            analysis.get("appointment_booked"),
            booking["booked"],
        )
        analysis["llm_booking_disagreement"] = True
    analysis["appointment_booked"] = booking["booked"]
    # Keep outcome and the booking flag consistent in both directions. A trace
    # reading outcome=appointment_booked with appointment_booked=False is worse
    # than either answer alone, because it looks like a data corruption bug.
    if booking["booked"]:
        analysis["outcome"] = "appointment_booked"
    elif analysis.get("outcome") == "appointment_booked":
        analysis["outcome"] = "needs_human_followup" if booking["attempted"] else "appointment_declined"
    analysis["appointment_details"] = booking["details"]
    analysis["telemetry"] = telemetry
    return analysis
