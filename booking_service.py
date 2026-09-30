"""A simulated clinic scheduling backend.

Stands in for a real EHR/scheduling integration. It is deterministic per call so
a demo replays identically, it persists confirmed bookings to a JSON file so the
result is inspectable after the call, and it deliberately has a slot that is
already taken so the agent has to handle a rejection rather than always
succeeding on the first attempt.
"""

from __future__ import annotations

import json
import os
import random
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

_LOCK = threading.Lock()
BOOKINGS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "bookings.json"
)

SPECIALTY_DOCTORS = {
    "Endocrinology": ["Dr. Anita Deshpande", "Dr. Marcus Whitfield"],
    "Internal Medicine": ["Dr. Leena Kapoor", "Dr. Samuel Osei"],
    "Cardiology": ["Dr. Priya Nair"],
}
_DEFAULT_DOCTORS = ["Dr. Leena Kapoor"]

# Slots the clinic never has free, so the agent must negotiate an alternative.
_UNAVAILABLE_HOURS = {8, 13, 18}


def _clinic_days(start_in_days: int = 1, count: int = 7) -> list[datetime]:
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    days = []
    d = start_in_days
    while len(days) < count:
        candidate = today + timedelta(days=d)
        if candidate.weekday() < 5:  # Monday to Friday
            days.append(candidate)
        d += 1
    return days


def get_availability(specialty: str, day_offset: int = 1, limit: int = 4) -> dict[str, Any]:
    """Return bookable slots, newest-first, for the requested specialty."""
    doctors = SPECIALTY_DOCTORS.get(specialty, _DEFAULT_DOCTORS)
    rng = random.Random(f"{specialty}:{day_offset}")
    slots: list[dict[str, str]] = []

    for day in _clinic_days(start_in_days=max(1, day_offset)):
        for hour in (9, 10, 11, 14, 15, 16, 17):
            if hour in _UNAVAILABLE_HOURS or rng.random() < 0.35:
                continue
            slots.append(
                {
                    "slot_id": f"{day.strftime('%Y%m%d')}-{hour:02d}00",
                    "date": day.strftime("%Y-%m-%d"),
                    "day_of_week": day.strftime("%A"),
                    "time": f"{hour % 12 or 12}:00 {'AM' if hour < 12 else 'PM'}",
                    "doctor": doctors[len(slots) % len(doctors)],
                    "specialty": specialty,
                    "mode": "in_person" if hour % 2 == 0 else "telehealth",
                }
            )
            if len(slots) >= limit:
                return {"specialty": specialty, "slots": slots, "count": len(slots)}
    return {"specialty": specialty, "slots": slots, "count": len(slots)}


def _load_bookings() -> list[dict[str, Any]]:
    if not os.path.exists(BOOKINGS_FILE):
        return []
    try:
        with open(BOOKINGS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return []


def _save_booking(record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(BOOKINGS_FILE), exist_ok=True)
    with _LOCK:
        records = _load_bookings()
        records.append(record)
        with open(BOOKINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)


def book(
    *,
    patient_id: str,
    patient_name: str,
    slot_id: str | None = None,
    date: str | None = None,
    time_of_day: str | None = None,
    specialty: str = "Internal Medicine",
    mode: str = "in_person",
    notes: str = "",
) -> dict[str, Any]:
    """Attempt to confirm an appointment.

    Returns a dict whose `status` is `confirmed` or `rejected`. The post-call
    analysis treats a `confirmed` status as the authoritative proof that an
    appointment was booked.
    """
    if not slot_id and not (date and time_of_day):
        return {
            "status": "rejected",
            "reason": "missing_slot",
            "message": "A slot id, or both a date and a time, are required to book.",
        }

    resolved_slot = slot_id or f"{date}-{time_of_day}"

    # Simulate a slot that was taken between the availability lookup and the booking.
    if slot_id and slot_id.endswith("-1200"):
        return {
            "status": "rejected",
            "reason": "slot_taken",
            "message": "That slot was just booked by someone else. Offer the patient another time.",
            "slot_id": slot_id,
        }

    for taken in _load_bookings():
        if taken.get("slot_id") == resolved_slot and taken.get("status") == "confirmed":
            return {
                "status": "rejected",
                "reason": "slot_taken",
                "message": "That slot is no longer available. Offer the patient another time.",
                "slot_id": resolved_slot,
            }

    doctors = SPECIALTY_DOCTORS.get(specialty, _DEFAULT_DOCTORS)
    confirmation = "APT-" + str(abs(hash(f"{patient_id}{resolved_slot}")) % 900000 + 100000)
    record = {
        "status": "confirmed",
        "confirmation_id": confirmation,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "slot_id": resolved_slot,
        "date": date,
        "time": time_of_day,
        "specialty": specialty,
        "doctor": doctors[0],
        "mode": mode,
        "notes": notes,
        "booked_at": datetime.now(timezone.utc).isoformat(),
        "message": (
            f"Appointment confirmed with {doctors[0]} ({specialty}) on {date} at {time_of_day}. "
            f"Confirmation number {confirmation}."
        ),
    }
    _save_booking(record)
    return record
