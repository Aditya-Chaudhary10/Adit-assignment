"""One-shot setup for real outbound telephony: Twilio SIP trunk to LiveKit.

Run once, then place real phone calls with `dispatch_call.py` (no --simulate).

    python setup_telephony.py --check              # what is configured already
    python setup_telephony.py --number +12025550123          # dry run
    python setup_telephony.py --number +12025550123 --apply  # actually create

Without --apply it prints exactly what it would create and changes nothing.

What it does
------------
1. Creates a Twilio Elastic SIP Trunk, or reuses one by name.
2. Creates a credential list with a generated username and password, so LiveKit
   authenticates with its own credential rather than your account password.
3. Associates your Twilio phone number with the trunk as the caller id.
4. Registers the trunk with LiveKit and prints SIP_OUTBOUND_TRUNK_ID.

Twilio credentials are read from the environment and never written to disk.
Prefer an API Key over your main auth token, and revoke it after the demo:
https://console.twilio.com/us1/account/keys-credentials/api-keys
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request

from livekit import api

from call_data import configure_console_encoding, load_environment

configure_console_encoding()
load_environment()

TWILIO_API = "https://api.twilio.com/2010-04-01"
TRUNKING_API = "https://trunking.twilio.com/v1"
TRUNK_NAME = os.getenv("TWILIO_TRUNK_NAME", "livekit-healthcare-agent")


def _auth_header() -> str:
    sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    key = os.getenv("TWILIO_API_KEY_SID") or sid
    secret = os.getenv("TWILIO_API_KEY_SECRET") or os.getenv("TWILIO_AUTH_TOKEN", "")
    if not sid or not secret:
        sys.exit(
            "Set TWILIO_ACCOUNT_SID and either TWILIO_AUTH_TOKEN, or "
            "TWILIO_API_KEY_SID plus TWILIO_API_KEY_SECRET."
        )
    token = base64.b64encode(f"{key}:{secret}".encode()).decode()
    return f"Basic {token}"


def twilio(method: str, url: str, data: dict[str, str] | None = None) -> dict:
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", _auth_header())
    if body:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:400]
        sys.exit(f"Twilio {method} failed with HTTP {exc.code}:\n{detail}")


def check() -> None:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    print(f"TWILIO_ACCOUNT_SID     : {'set (' + sid[:8] + '...)' if sid else 'MISSING'}")
    print(f"TWILIO_AUTH_TOKEN      : {'set' if os.getenv('TWILIO_AUTH_TOKEN') else 'missing'}")
    print(f"SIP_OUTBOUND_TRUNK_ID  : {os.getenv('SIP_OUTBOUND_TRUNK_ID') or 'not set yet'}")
    if not sid:
        return

    acct = twilio("GET", f"{TWILIO_API}/Accounts/{sid}.json")
    print(f"\nTwilio account         : {acct.get('friendly_name')} | type={acct.get('type')}")
    if str(acct.get("type", "")).lower() == "trial":
        print("  TRIAL ACCOUNT: you can only call numbers verified in the console.")

    nums = twilio("GET", f"{TWILIO_API}/Accounts/{sid}/IncomingPhoneNumbers.json")
    owned = nums.get("incoming_phone_numbers", [])
    print(f"\nPhone numbers owned    : {len(owned)}")
    for n in owned:
        voice = (n.get("capabilities") or {}).get("voice")
        print(f"  {n.get('phone_number')}  voice={voice}  ({n.get('friendly_name')})")
    if not owned:
        print("  Buy a voice-capable number in the Twilio console first.")

    trunks = twilio("GET", f"{TRUNKING_API}/Trunks").get("trunks", [])
    print(f"\nSIP trunks             : {len(trunks)}")
    for t in trunks:
        print(f"  {t.get('friendly_name')} -> {t.get('domain_name')} (sid {t.get('sid')})")


def ensure_trunk(apply: bool) -> dict:
    """Find or create the trunk, without ever deleting one.

    Twilio trial accounts allow exactly one SIP trunk. If the account already has
    one, reuse it rather than deleting whatever is there: it may belong to
    another project. Everything this script adds to a reused trunk is additive.
    """
    trunks = twilio("GET", f"{TRUNKING_API}/Trunks").get("trunks", [])
    for t in trunks:
        if t.get("friendly_name") == TRUNK_NAME:
            print(f"[=] Reusing our trunk {t['sid']} ({t['domain_name']})")
            return t

    if trunks:
        t = trunks[0]
        print(
            f"[=] Trial accounts allow one trunk and this account already has "
            f"'{t.get('friendly_name')}'. Reusing it rather than deleting it."
        )
        print(f"    {t['sid']} -> {t['domain_name']}")
        return t

    domain = f"{TRUNK_NAME}-{secrets.token_hex(3)}.pstn.twilio.com"
    if not apply:
        print(f"[+] WOULD create Twilio trunk '{TRUNK_NAME}' at {domain}")
        return {"sid": "<pending>", "domain_name": domain}
    t = twilio(
        "POST", f"{TRUNKING_API}/Trunks", {"FriendlyName": TRUNK_NAME, "DomainName": domain}
    )
    print(f"[+] Created Twilio trunk {t['sid']} ({t['domain_name']})")
    return t


def ensure_credentials(trunk_sid: str, apply: bool) -> tuple[str, str]:
    username = f"lk{secrets.token_hex(5)}"
    password = secrets.token_urlsafe(24)
    if not apply:
        print(f"[+] WOULD create credential list '{TRUNK_NAME}' with user {username}")
        return username, password

    sid = os.environ["TWILIO_ACCOUNT_SID"]
    cl = twilio(
        "POST",
        f"{TWILIO_API}/Accounts/{sid}/SIP/CredentialLists.json",
        {"FriendlyName": TRUNK_NAME},
    )
    twilio(
        "POST",
        f"{TWILIO_API}/Accounts/{sid}/SIP/CredentialLists/{cl['sid']}/Credentials.json",
        {"Username": username, "Password": password},
    )
    twilio(
        "POST",
        f"{TRUNKING_API}/Trunks/{trunk_sid}/CredentialLists",
        {"CredentialListSid": cl["sid"]},
    )
    print(f"[+] Created credential list {cl['sid']} with user {username}")
    return username, password


def attach_number(trunk_sid: str, number: str, apply: bool) -> None:
    sid = os.environ["TWILIO_ACCOUNT_SID"]
    nums = twilio("GET", f"{TWILIO_API}/Accounts/{sid}/IncomingPhoneNumbers.json")
    match = next(
        (n for n in nums.get("incoming_phone_numbers", []) if n.get("phone_number") == number),
        None,
    )
    if not match:
        sys.exit(f"{number} is not owned by this Twilio account. Buy it first.")
    attached = twilio("GET", f"{TRUNKING_API}/Trunks/{trunk_sid}/PhoneNumbers").get(
        "phone_numbers", []
    )
    if any(a.get("phone_number") == number for a in attached):
        print(f"[=] {number} is already on the trunk")
        return
    if not apply:
        print(f"[+] WOULD attach {number} ({match['sid']}) to the trunk")
        return
    twilio(
        "POST",
        f"{TRUNKING_API}/Trunks/{trunk_sid}/PhoneNumbers",
        {"PhoneNumberSid": match["sid"]},
    )
    print(f"[+] Attached {number} to the trunk")


async def register_with_livekit(
    address: str, number: str, user: str, pwd: str, apply: bool
) -> None:
    if not apply:
        print(f"[+] WOULD register a LiveKit outbound trunk to {address} as {number}")
        return
    lk = api.LiveKitAPI()
    try:
        info = await lk.sip.create_outbound_trunk(
            api.CreateSIPOutboundTrunkRequest(
                trunk=api.SIPOutboundTrunkInfo(
                    name=TRUNK_NAME,
                    address=address,
                    numbers=[number],
                    auth_username=user,
                    auth_password=pwd,
                )
            )
        )
        print(f"\n[+] LiveKit outbound trunk created: {info.sip_trunk_id}")
        print("\nAdd this line to .env:")
        print(f"    SIP_OUTBOUND_TRUNK_ID={info.sip_trunk_id}")
        print("\nThen place a real call:")
        print("    python dispatch_call.py --patient PT-10432 --phone +91XXXXXXXXXX")
    finally:
        await lk.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="Report current state and exit")
    parser.add_argument("--number", help="Your Twilio number in E.164, e.g. +12025550123")
    parser.add_argument("--apply", action="store_true", help="Actually create things")
    args = parser.parse_args()

    if args.check or not args.number:
        check()
        if not args.number:
            print("\nPass --number +1... to set up the trunk. Add --apply to create it.")
        return 0

    if not args.apply:
        print("DRY RUN. Nothing will be created. Re-run with --apply.\n")

    trunk = ensure_trunk(args.apply)
    user, pwd = ensure_credentials(trunk["sid"], args.apply)
    attach_number(trunk["sid"], args.number, args.apply)
    asyncio.run(register_with_livekit(trunk["domain_name"], args.number, user, pwd, args.apply))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
