"""Print a browser link that joins a simulated call room.

    python join_link.py                      # newest active room
    python join_link.py <room-name>          # a specific room
    python join_link.py --list               # show active rooms

Only needed for `--simulate` calls, where a human stands in for the patient.
A real outbound call reaches an actual phone and needs none of this.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import urllib.parse

from livekit import api

from call_data import configure_console_encoding, load_environment

configure_console_encoding()
load_environment()


async def active_rooms() -> list[str]:
    lkapi = api.LiveKitAPI()
    try:
        rooms = await lkapi.room.list_rooms(api.ListRoomsRequest())
        return [r.name for r in rooms.rooms]
    finally:
        await lkapi.aclose()


def build_link(room: str, identity: str = "patient-browser") -> str:
    token = (
        api.AccessToken()
        .with_identity(identity)
        .with_name("Patient")
        .with_grants(
            api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)
        )
        .to_jwt()
    )
    url = os.environ["LIVEKIT_URL"]
    return (
        "https://meet.livekit.io/custom?liveKitUrl="
        + urllib.parse.quote(url, safe="")
        + "&token="
        + urllib.parse.quote(token, safe="")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("room", nargs="?", help="Room name; defaults to the newest active room")
    parser.add_argument("--list", action="store_true", help="List active rooms and exit")
    args = parser.parse_args()

    if not os.getenv("LIVEKIT_URL"):
        sys.exit("LIVEKIT_URL is not set. Fill in .env first.")

    rooms = asyncio.run(active_rooms())

    if args.list:
        print("\n".join(rooms) if rooms else "(no active rooms)")
        return 0

    room = args.room
    if not room:
        outbound = [r for r in rooms if r.startswith("outbound-")]
        if not outbound:
            sys.exit(
                "No active outbound room found.\n"
                "Start one with:  python dispatch_call.py --patient PT-10432 --simulate"
            )
        room = sorted(outbound)[-1]

    print(f"Room: {room}\n")
    print(build_link(room))
    print("\nOpen that in a browser and allow the microphone. The agent speaks first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
