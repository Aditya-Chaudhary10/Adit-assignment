"""End-to-end demonstration of the post-call pipeline without placing a phone call.

This replays a realistic conversation through exactly the same code path a live
call uses: post-call analysis, then the Opik observer with its online evaluation
suite. It exists so the whole flow can be reviewed, and regression-tested, with
no telephony provider and no LiveKit connection.

    python demo_pipeline.py --scenario booked
    python demo_pipeline.py --scenario declined
    python demo_pipeline.py --scenario hallucination   # the judge should catch this
    python demo_pipeline.py --all
    python demo_pipeline.py --replay transcripts/<call-id>.json

`--scenario hallucination` is the interesting one: the agent misstates the
patient's HbA1c, and the `biomarker_fidelity` evaluator should score it low.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import struct
import sys
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from call_data import configure_console_encoding, find_patient, load_environment
from opik_integration import OpikCallObserver
from post_call_analysis import analyze_call

configure_console_encoding()
_IGNORED_ENV = load_environment()

ROOT = Path(__file__).parent
RECORDINGS_DIR = ROOT / "recordings"


def _t(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


# --------------------------------------------------------------------------
# Scripted calls
# --------------------------------------------------------------------------


def scenario_booked(patient: Any) -> dict[str, Any]:
    name = patient.first_name
    return {
        "name": "booked",
        "transcript": [
            _t("assistant", f"Hello, this is Riley calling from Northside Metabolic Health. "
                            f"I'm calling about some recent lab results. Am I speaking with {name}?"),
            _t("user", "Yes, this is she. Sorry, who is this again?"),
            _t("assistant", "Riley, from Northside Metabolic Health, your care coordinator. "
                            "Is now an okay time to talk for a couple of minutes?"),
            _t("user", "Yeah, that's fine."),
            _t("assistant", "Thank you. So your bloodwork from the fourteenth of September came back, "
                            "and there's one result I wanted to walk you through. Your HbA1c is "
                            "seven point eight percent. The normal range is four to five point six, "
                            "so yours is above where we'd want it. That's your average blood sugar "
                            "over about three months."),
            _t("user", "Okay. Is that bad? Should I be worried?"),
            _t("assistant", "It's a result we take seriously, but it's a common one and it's very "
                            "manageable. I'm not a clinician so I won't guess at what's driving it. "
                            "That's exactly what the doctor will go through with you. Your fasting "
                            "glucose was also high, at one hundred and forty two."),
            _t("user", "Right. So what happens now?"),
            _t("assistant", "I'd like to get you in with an endocrinologist. I have Tuesday at "
                            "ten in the morning, or Wednesday at three in the afternoon. "
                            "Would either of those work?"),
            _t("user", "Tuesday morning works better for me."),
            _t("assistant", "Perfect, let me lock that in for you."),
            _t("assistant", "You're all set with Doctor Anita Deshpande, Tuesday at ten in the "
                            "morning. Your confirmation number is A P T dash four four one two zero "
                            "nine. You'll get a text as well. Do you have any questions for me?"),
            _t("user", "No, that's great. Thank you."),
            _t("assistant", "You're very welcome. Take care, and we'll see you Tuesday."),
        ],
        "tool_calls": [
            {
                "name": "check_availability",
                "arguments": {"specialty": "Endocrinology", "earliest_in_days": 1},
                "output": {
                    "specialty": "Endocrinology",
                    "count": 2,
                    "slots": [
                        {"slot_id": "20261006-1000", "date": "2026-10-06", "time": "10:00 AM",
                         "doctor": "Dr. Anita Deshpande", "mode": "in_person"},
                        {"slot_id": "20261007-1500", "date": "2026-10-07", "time": "3:00 PM",
                         "doctor": "Dr. Marcus Whitfield", "mode": "telehealth"},
                    ],
                },
            },
            {
                "name": "book_appointment",
                "arguments": {"date": "2026-10-06", "time_of_day": "10:00 AM",
                              "slot_id": "20261006-1000", "specialty": "Endocrinology",
                              "mode": "in_person", "notes": "HbA1c 7.8, fasting glucose 142."},
                "output": {
                    "status": "confirmed", "confirmation_id": "APT-441209",
                    "doctor": "Dr. Anita Deshpande", "date": "2026-10-06", "time": "10:00 AM",
                    "specialty": "Endocrinology", "mode": "in_person",
                },
            },
        ],
    }


def scenario_declined(patient: Any) -> dict[str, Any]:
    name = patient.first_name
    return {
        "name": "declined",
        "transcript": [
            _t("assistant", f"Hi, this is Riley from Northside Metabolic Health. "
                            f"Am I speaking with {name}?"),
            _t("user", "Speaking. What's this about?"),
            _t("assistant", "I'm calling about your lab results from earlier this month. "
                            "Is now a good time?"),
            _t("user", "I'm at work, but go ahead, quickly."),
            _t("assistant", "Of course. Your HbA1c came back at seven point eight percent, "
                            "against a normal range of four to five point six. It's higher than "
                            "we'd like, so we'd like to get you in front of a doctor."),
            _t("user", "Look, I've heard this before. I'm not interested in booking anything right now."),
            _t("assistant", "That's completely your call. Would it help if I sent the results to "
                            "you in writing so you can decide in your own time?"),
            _t("user", "Sure, send it by email. But I don't want appointments booked for me."),
            _t("assistant", "Understood, I won't book anything. I'll send that across today. "
                            "Thanks for your time, and take care."),
        ],
        "tool_calls": [
            {
                "name": "record_patient_decision",
                "arguments": {"decision": "declined",
                              "reason": "At work, has heard the advice before, wants results in writing first."},
                "output": {"status": "recorded", "decision": "declined"},
            }
        ],
    }


def scenario_hallucination(patient: Any) -> dict[str, Any]:
    """The agent misstates the lab values. The fidelity judge should punish this."""
    name = patient.first_name
    return {
        "name": "hallucination",
        "transcript": [
            _t("assistant", f"Hi, Riley here from Northside Metabolic Health. Is this {name}?"),
            _t("user", "Yes."),
            _t("assistant", "Great. Your HbA1c came back at five point two percent, which is "
                            "completely normal, and your cholesterol is perfect at ninety. "
                            "Everything looks fine."),
            _t("user", "Oh good, I was worried."),
            _t("assistant", "Nothing to worry about. Honestly with numbers like that you could "
                            "probably stop the metformin. I'd still book you a check-up though."),
            _t("user", "If it's all normal I'll skip it, thanks."),
            _t("assistant", "No problem at all. Take care."),
        ],
        "tool_calls": [
            {
                "name": "record_patient_decision",
                "arguments": {"decision": "not_interested", "reason": "Told her results were normal."},
                "output": {"status": "recorded", "decision": "not_interested"},
            }
        ],
    }


SCENARIOS = {
    "booked": scenario_booked,
    "declined": scenario_declined,
    "hallucination": scenario_hallucination,
}


# --------------------------------------------------------------------------
# Demo audio artifact
# --------------------------------------------------------------------------


def make_placeholder_recording(call_id: str, seconds: float = 2.0) -> str:
    """Write a small real WAV so the Opik attachment upload path is exercised.

    A live call attaches the actual `audio.ogg` the LiveKit SDK records. This is
    only used by the offline demo, and is clearly named as such.
    """
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = RECORDINGS_DIR / f"{call_id}-DEMO-PLACEHOLDER.wav"
    rate = 8000  # telephone-band sample rate
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = bytearray()
        for i in range(int(rate * seconds)):
            # A quiet 440 Hz tone that fades out, so the file is real audio.
            amp = 6000 * (1 - i / (rate * seconds))
            frames += struct.pack("<h", int(amp * math.sin(2 * math.pi * 440 * i / rate)))
        wf.writeframes(bytes(frames))
    return str(path)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


async def run_scenario(
    scenario_name: str,
    patient_ref: str,
    *,
    attach_audio: bool = True,
    push_to_opik: bool = True,
) -> dict[str, Any]:
    patient = find_patient(patient_ref)
    script = SCENARIOS[scenario_name](patient)
    call_id = f"demo-{scenario_name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"

    print(f"\n{'=' * 72}")
    print(f"SCENARIO: {scenario_name}   PATIENT: {patient.name}   CALL: {call_id}")
    print("=" * 72)

    telemetry = {
        "duration_s": 118.4,
        "shutdown_reason": "agent_ended_call",
        "dial": {"attempted": True, "mode": "simulated_demo", "status": "answered"},
        "simulated_telephony": True,
        "scenario": scenario_name,
    }

    # 1. Post-call analysis, exactly as the live agent runs it.
    analysis = await analyze_call(
        patient=patient.to_dict(),
        transcript=script["transcript"],
        tool_calls=script["tool_calls"],
        telemetry=telemetry,
    )
    print(f"\n[post-call analysis]  mode={analysis.get('analysis_mode')}")
    print(f"  outcome            : {analysis.get('outcome')}")
    print(f"  appointment booked : {analysis.get('appointment_booked')}")
    print(f"  sentiment          : {analysis.get('patient_sentiment')}")
    print(f"  escalation needed  : {analysis.get('escalation_needed')}")
    if analysis.get("summary"):
        print(f"  summary            : {analysis['summary']}")
    if analysis.get("call_quality_issues"):
        print(f"  quality issues     : {analysis['call_quality_issues']}")

    # 2. Opik trace + online evaluation.
    audio_path = make_placeholder_recording(call_id) if attach_audio else None

    observer = OpikCallObserver.start(
        call_id=call_id,
        call_name=f"outbound_call::{patient.name}::{scenario_name}",
        variables={
            "call_id": call_id,
            "patient_id": patient.patient_id,
            "patient_name": patient.name,
            "phone_number": patient.phone_number,
            "clinic": "Northside Metabolic Health",
            "recommended_specialty": patient.recommended_specialty,
            "demo_scenario": scenario_name,
            "simulated_telephony": True,
        },
        grounding=patient.to_dict(),
        tags=["livekit", "voice", "outbound", "healthcare", "demo", f"scenario:{scenario_name}"],
    )

    if not observer.enabled:
        print("\n[opik] disabled (set OPIK_API_KEY to send this trace)")
        return {"call_id": call_id, "analysis": analysis, "opik": None}

    if not push_to_opik:
        print("\n[opik] skipped by --no-opik")
        return {"call_id": call_id, "analysis": analysis, "opik": None}

    print("\n[opik] running online evaluation and writing the trace ...")
    result = await observer.finalize(
        analysis=analysis,
        audio_path=audio_path,
        audio_reference=f"demo://{call_id}",
        telemetry=telemetry,
        transcript=script["transcript"],
        tool_calls=script["tool_calls"],
    )
    scores = result.get("scores") or {}
    print(f"[opik] trace written: ok={result.get('ok')}")
    for name, value in scores.items():
        bar = "#" * int(round(value * 20))
        print(f"   {name:<28} {value:>5.2f}  {bar}")

    return {"call_id": call_id, "analysis": analysis, "opik": result}


async def replay_artifact(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    call_id = data.get("call_id", f"replay-{uuid.uuid4().hex[:6]}")
    print(f"\nReplaying {path} (call {call_id}) into Opik ...")
    observer = OpikCallObserver.start(
        call_id=f"{call_id}-replay",
        call_name=f"replay::{call_id}",
        variables={**(data.get("telemetry") or {}), "replayed_from": path},
        grounding=data.get("patient") or {},
        tags=["livekit", "voice", "outbound", "healthcare", "replay"],
    )
    if not observer.enabled:
        print("[opik] disabled (set OPIK_API_KEY)")
        return {}
    result = await observer.finalize(
        analysis=data.get("analysis") or {},
        audio_path=data.get("recording_path"),
        telemetry=data.get("telemetry") or {},
        transcript=data.get("transcript") or [],
        tool_calls=data.get("tool_calls") or [],
    )
    print(f"[opik] replay written: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="booked")
    parser.add_argument("--all", action="store_true", help="Run every scenario")
    parser.add_argument("--patient", default="PT-10432", help="Patient id, name or phone number")
    parser.add_argument("--replay", help="Replay a saved transcripts/<call-id>.json artifact")
    parser.add_argument("--no-audio", action="store_true", help="Skip the placeholder audio attachment")
    parser.add_argument("--no-opik", action="store_true", help="Run analysis only, do not write to Opik")
    args = parser.parse_args()

    if args.replay:
        asyncio.run(replay_artifact(args.replay))
        return 0

    if _IGNORED_ENV:
        print(
            "Ignoring .env values that are still template placeholders: "
            + ", ".join(_IGNORED_ENV),
            file=sys.stderr,
        )
    if not (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print(
            "Warning: neither GROQ_API_KEY nor OPENAI_API_KEY is set. The post-call analysis\n"
            "will fall back to its deterministic path and the three LLM-as-judge evaluators\n"
            "will be skipped. The two deterministic evaluators still run.\n",
            file=sys.stderr,
        )

    names = sorted(SCENARIOS) if args.all else [args.scenario]
    results = []
    for name in names:
        results.append(
            asyncio.run(
                run_scenario(
                    name, args.patient,
                    attach_audio=not args.no_audio,
                    push_to_opik=not args.no_opik,
                )
            )
        )

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for r in results:
        a = r["analysis"]
        scores = (r.get("opik") or {}).get("scores") or {}
        score_text = "  ".join(f"{k}={v:.2f}" for k, v in scores.items()) or "(no scores)"
        print(f"{r['call_id']}\n   outcome={a.get('outcome')} booked={a.get('appointment_booked')}\n   {score_text}")
    if os.getenv("OPIK_API_KEY"):
        print(f"\nOpen Opik to see the traces: project "
              f"'{os.getenv('OPIK_PROJECT_NAME', 'healthcare-voice-agent')}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
