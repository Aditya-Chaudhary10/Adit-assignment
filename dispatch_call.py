"""Place an outbound call by dispatching the agent worker to a fresh room.

    python dispatch_call.py --patient PT-10432
    python dispatch_call.py --patient "Meera" --phone +14155550123
    python dispatch_call.py --patient PT-10432 --simulate     # no telephony
    python dispatch_call.py --list

The worker (`agent.py`) must already be running. The patient record travels to
the worker as the dispatch metadata, so the agent knows who it is calling and
which biomarkers to discuss before it ever picks up the line.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone

from livekit import api

from call_data import (
    CallContext,
    configure_console_encoding,
    find_patient,
    load_environment,
    load_patients,
)

configure_console_encoding()
_IGNORED_ENV = load_environment()

AGENT_NAME = os.getenv("AGENT_NAME", "healthcare-outbound-agent")

REQUIRED_ENV = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")


def _check_env() -> None:
    if _IGNORED_ENV:
        print(
            "Ignoring .env values that are still template placeholders: "
            + ", ".join(_IGNORED_ENV),
            file=sys.stderr,
        )
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        sys.exit(
            f"Missing required environment variables: {', '.join(missing)}.\n"
            "Copy .env.example to .env and fill in your LiveKit project credentials."
        )


def _list_patients() -> None:
    print(f"{'PATIENT ID':<12} {'NAME':<22} {'PHONE':<18} SPECIALTY")
    print("-" * 78)
    for p in load_patients():
        flags = ", ".join(b.key for b in p.abnormal_biomarkers) or "none"
        print(f"{p.patient_id:<12} {p.name:<22} {p.phone_number:<18} {p.recommended_specialty}")
        print(f"{'':<12} out of range: {flags}")


async def place_call(args: argparse.Namespace) -> int:
    patient = find_patient(args.patient)
    if args.phone:
        patient.phone_number = args.phone

    call_id = args.call_id or f"call-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    room_name = args.room or f"outbound-{call_id}"

    call = CallContext(
        patient=patient,
        call_id=call_id,
        clinic_name=args.clinic,
        agent_display_name=args.agent_persona,
        transfer_to=args.transfer_to,
        simulate=args.simulate,
    )

    print(f"Call id     : {call_id}")
    print(f"Room        : {room_name}")
    print(f"Patient     : {patient.name} ({patient.patient_id})")
    print(f"Calling     : {patient.phone_number if not args.simulate else '(simulated, no telephony)'}")
    print(f"Out of range: {', '.join(b.label for b in patient.abnormal_biomarkers) or 'none'}")
    print(f"Booking for : {patient.recommended_specialty}")
    print()

    lkapi = api.LiveKitAPI()
    try:
        # Create the room explicitly so we control how long it survives with
        # nobody in it. A real outbound call fills the room within seconds, but
        # in simulation a human has to open a browser and join, and the default
        # empty-room timeout closes the room out from under them first.
        empty_timeout = args.empty_timeout if args.simulate else 60
        await lkapi.room.create_room(
            api.CreateRoomRequest(name=room_name, empty_timeout=empty_timeout)
        )

        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=call.to_metadata(),
            )
        )
        print(f"Dispatched agent '{AGENT_NAME}' (dispatch id {dispatch.id}).")
        if args.simulate:
            print(
                "\nSimulation mode: no phone call is placed.\n"
                f"Join room '{room_name}' to talk to the agent, for example at\n"
                "  https://agents-playground.livekit.io  (select this room)\n"
            )
        else:
            print("\nThe agent is dialing now. Watch the worker logs for progress.")
        print(f"After the call, the trace appears in Opik project "
              f"'{os.getenv('OPIK_PROJECT_NAME', 'healthcare-voice-agent')}'.")
        print(f"Local artifact: transcripts/{call_id}.json")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to dispatch the agent: {exc}", file=sys.stderr)
        print(
            "\nCheck that:\n"
            "  1. the worker is running    (python agent.py dev)\n"
            f"  2. its agent_name matches  ({AGENT_NAME})\n"
            "  3. LIVEKIT_URL / API key / secret are correct",
            file=sys.stderr,
        )
        return 1
    finally:
        await lkapi.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Place an outbound healthcare voice call via a LiveKit agent."
    )
    parser.add_argument("--patient", help="Patient id, phone number, or name from data/patients.json")
    parser.add_argument("--phone", help="Override the phone number on the patient record")
    parser.add_argument("--room", help="Room name (defaults to outbound-<call id>)")
    parser.add_argument("--call-id", help="Explicit call id (defaults to a timestamped id)")
    parser.add_argument("--transfer-to", help="Number to transfer to when the patient asks for a human")
    parser.add_argument("--clinic", default="Northside Metabolic Health", help="Clinic name the agent uses")
    parser.add_argument("--agent-persona", default="Riley", help="Name the agent introduces itself with")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Skip telephony and run the agent in the room so you can join from a browser",
    )
    parser.add_argument(
        "--empty-timeout",
        type=int,
        default=1800,
        help="Seconds the simulated room stays open with nobody in it (default 1800)",
    )
    parser.add_argument("--list", action="store_true", help="List the patients on file and exit")
    args = parser.parse_args()

    if args.list:
        _list_patients()
        return 0
    if not args.patient:
        parser.error("--patient is required (or use --list)")

    _check_env()
    return asyncio.run(place_call(args))


if __name__ == "__main__":
    raise SystemExit(main())
