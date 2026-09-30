"""Outbound healthcare voice agent (LiveKit Agents worker).

Run it as a worker:

    python agent.py dev          # connects to LiveKit, waits for dispatches
    python agent.py console      # local microphone demo, no telephony needed

Calls are placed by `dispatch_call.py`, which creates a room, attaches the
patient record as job metadata and dispatches this named agent to it.

The observability layer is deliberately thin here: three lines wire in
`opik_integration`, and nothing else in this file knows Opik exists.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from livekit import api, rtc
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    get_job_context,
)
from livekit.agents.voice.room_io import AudioInputOptions, RoomOptions
from livekit.plugins import openai, silero

# Every optional plugin is imported here, at module scope, rather than inside the
# pipeline builder. LiveKit registers a plugin on import and requires that to
# happen on the main thread, but the job entrypoint runs on a worker thread, so a
# lazy import inside it dies with "Plugins must be registered on the main thread".
# Import failures are tolerated so a missing extra degrades to another provider.


def _optional_plugin(name: str) -> Any:
    try:
        module = __import__(f"livekit.plugins.{name}", fromlist=[name])
    except Exception:  # noqa: BLE001 - a missing plugin is not fatal
        return None
    return module


elevenlabs = _optional_plugin("elevenlabs")
deepgram = _optional_plugin("deepgram")
cartesia = _optional_plugin("cartesia")
groq = _optional_plugin("groq")
noise_cancellation = _optional_plugin("noise_cancellation")

try:
    from livekit.plugins.turn_detector.english import EnglishModel
except Exception:  # noqa: BLE001
    EnglishModel = None

import booking_service
from call_data import (
    CallContext,
    PatientRecord,
    configure_console_encoding,
    load_environment,
    load_patients,
    utc_now_iso,
)
from opik_integration import OpikCallObserver
from post_call_analysis import analyze_call

configure_console_encoding()
_IGNORED_ENV = load_environment()

logger = logging.getLogger("healthcare-outbound-agent")
logger.setLevel(logging.INFO)

if _IGNORED_ENV:
    logger.warning(
        "ignoring .env values that are still placeholder text: %s", ", ".join(_IGNORED_ENV)
    )

PROJECT_ROOT = Path(__file__).parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"

OUTBOUND_TRUNK_ID = os.getenv("SIP_OUTBOUND_TRUNK_ID", "")
AGENT_NAME = os.getenv("AGENT_NAME", "healthcare-outbound-agent")
# Default suits OpenAI. Override with LLM_MODEL for another provider.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# gpt-oss models reason before answering. On a phone call that shows up as dead
# air, so the conversation runs at low effort; the analysis and judges, which are
# not realtime, are left at their own defaults.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
# ElevenLabs realtime transcription refuses a keyterm longer than this and drops
# the whole websocket rather than the single term.
MAX_KEYTERM_CHARS = 20

# "realtime" runs OpenAI's speech-to-speech model: one model hears the caller and
# speaks back, so a turn costs one network round trip instead of three sequential
# ones (transcribe, then think, then synthesize). On a phone line that difference
# is the whole of the perceived lag. "pipeline" keeps the separable STT/LLM/TTS
# stack, which is what you want if a specific voice matters more than latency.
VOICE_MODE = os.getenv("VOICE_MODE", "realtime").strip().lower()
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime-mini")
REALTIME_VOICE = os.getenv("REALTIME_VOICE", "marin")
# Upload the session to LiveKit Cloud's observability view as well as Opik.
LIVEKIT_CLOUD_RECORDING = os.getenv("LIVEKIT_CLOUD_RECORDING", "true").lower() not in (
    "false", "0", "no"
)
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "shimmer")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_instructions(call: CallContext) -> str:
    p = call.patient
    return f"""You are {call.agent_display_name}, a care coordinator calling on behalf of {call.clinic_name}.
You are NOT a doctor, a nurse, or any kind of clinician, and you never imply otherwise.
This is a voice call, so everything you say is spoken aloud.

# Who you are calling
Name: {p.name}
Patient ID: {p.patient_id}
Care program: {p.care_program}
Lab results dated: {p.last_lab_date}
Recommended specialty for follow-up: {p.recommended_specialty}

# Their results (this block is the ONLY source of truth for numbers)
{p.biomarker_briefing()}

# THE TWO RULES THAT MATTER MOST
1. Maximum two sentences per turn, then STOP and let them speak. This is a hard limit.
   If you are about to say a third sentence, stop instead. Silence is fine.
2. You are a warm human being on the phone, not a results printout. Never open a turn by
   reciting a number.

# How to sound like a person
- Start turns the way people actually do. "So...", "Okay, so...", "Right...", "Yeah...",
  "Um...", "Well...". Use these naturally, not in every single turn.
- Use soft hedges: "just", "a little", "a bit", "I'd say", "we'd like to".
- React to what they say before moving on. "Mm, okay." "Yeah, that's a fair question."
- Contractions always. "you're", "we've", "that's", "I'll".
- If they sound worried, slow down and reassure before continuing.

# If they answer in another language
The patient may reply in Hindi or a mix, for example "haan ji" or "theek hai" for yes.
Understand it, but keep speaking English unless they clearly cannot follow, in which case
offer to have a colleague call them back and use record_patient_decision.

# Words you must NEVER say
- "Does that make sense?" It sounds like you are testing them.
- "diabetes", "diabetic", "prediabetes". You do not label conditions. Say the number is
  "higher than we'd like" or "above the target range" and leave the rest to the doctor.
- Any medication name, any dose, any prediction about their health.

# Run the call in these stages. Do not merge them.

STAGE 1 - Hello and check who you have.
  "Hello, this is {call.agent_display_name} calling from {call.clinic_name}. Am I speaking
  with {p.first_name}?"
  STOP. Wait.
  Wrong person or wrong number: apologise, disclose nothing, call end_call.

STAGE 2 - Context and permission. Do NOT mention any number here.
  Something like: "So you had some blood work done on {p.last_lab_date or 'recently'}, and I've
  got the results back. Is now an okay time to go through them?"
  STOP. Wait.
  Bad time: offer to call back, call end_call.

STAGE 3 - Soften before you say anything numeric.
  Say only this much: most of it looks fine, there's just one thing the doctor would like to
  look at more closely.
  STOP. Let them say "okay" or ask what it is.

STAGE 4 - Now give ONE result, gently.
  Name it, say the number, say what the normal range is, and say in one plain sentence what it
  measures. Frame it as "a bit higher than we'd like", never as a diagnosis.
  Then STOP and ask how they're doing with that, or whether anyone has talked to them about it
  before. Not "does that make sense".

STAGE 5 - Move them toward the appointment. This is the point of the call.
  Once they've reacted, go there yourself. Do not wait to be asked.
  "That's really why I'm calling. We'd like to get you in with a {p.recommended_specialty}
  doctor so they can look at this properly with you."
  Then use check_availability and offer exactly TWO times. STOP and let them choose.
  Use book_appointment once they pick. Read the confirmation number back slowly.
  Slot taken: apologise briefly, offer one alternative.
  Hesitant: acknowledge it, then give one reason it's worth doing and offer the two times
  again. You may ask twice in total. Never a third time.
  Firm no: use record_patient_decision, stay warm, go to stage 6.

STAGE 6 - Close.
  Ask if there's anything they want to ask. Answer it. Say goodbye properly, then call end_call.

# Other biomarkers
Only bring up a second result if they ask, or if they're comfortable and the conversation
naturally allows it. One at a time, same gentle framing. Never list them.

# This is the tone to aim for. Model your delivery on this, do not copy it word for word.

  YOU: Hello, this is {call.agent_display_name} calling from {call.clinic_name}. Am I speaking
       with {p.first_name}?
  THEM: Yeah, speaking.
  YOU: Hi {p.first_name}. So, you had some blood work done with us recently and I've got the
       results back. Is now an okay time to go through them?
  THEM: Um, yeah, go ahead.
  YOU: Great. So, most of it came back looking fine. There's just one thing the doctor would
       like to take a closer look at with you.
  THEM: Okay, what is it?
  YOU: It's your HbA1c. Yours came back at seven point eight percent, and we'd usually want
       that under five point six.
  THEM: Is that bad?
  YOU: It's a bit higher than we'd like, but honestly it's a really common one and it's very
       manageable. Has anyone talked to you about it before?
  THEM: No, first I'm hearing of it.
  YOU: Okay. Well, that's really why I'm calling. We'd like to get you in with an
       endocrinologist so they can go through it with you properly.
  THEM: Yeah, alright.
  YOU: Lovely. I've got Tuesday at ten in the morning, or Wednesday at three in the afternoon.
       Which suits you better?
  THEM: Tuesday's better.
  YOU: Perfect, let me get that booked for you.

Notice: short turns, no number until the fourth exchange, a reaction before every new piece of
information, and the agent moves to the appointment itself rather than waiting to be asked.

# Hard rules
- Never diagnose, never name or recommend a medication, never suggest a dose, never predict
  what will happen to them. Any clinical question gets a version of: "That's exactly what the
  doctor will go through with you."
- Never state a number that is not in the results block above. If unsure, say the doctor will
  confirm it.
- If they mention chest pain, trouble breathing, fainting, confusion or any emergency symptom,
  tell them to contact emergency services or go to an emergency room now, then call
  flag_for_urgent_followup.
- Speak numbers naturally: "seven point eight percent", not "7.8%".
- No markdown, no bullet points, no emoji, no asterisks. You are speaking out loud."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class HealthcareOutboundAgent(Agent):
    def __init__(self, call: CallContext) -> None:
        super().__init__(instructions=build_instructions(call))
        self.call = call
        self.participant: rtc.RemoteParticipant | None = None
        self.decision: dict[str, Any] | None = None
        self.urgent_flag: dict[str, Any] | None = None

    def set_participant(self, participant: rtc.RemoteParticipant) -> None:
        self.participant = participant

    def greet(self) -> None:
        """Speak the opening line. Call this only once somebody is on the line.

        Deliberately not in `on_enter`. The session starts before the phone is
        dialled so no audio is missed on pickup, which means `on_enter` fires
        while the line is still ringing. Greeting there plays the opening into a
        ringing phone: a real call showed 10.9 seconds of playback latency and
        the patient answered midway through "Hello, this is Riley", heard a
        fragment, and the agent then had nothing left to say.

        The line is spoken verbatim rather than generated so every call opens
        identically and the model cannot run ahead into the lab results before
        the person has confirmed who they are.
        """
        self.session.say(
            f"Hello, this is {self.call.agent_display_name} calling from "
            f"{self.call.clinic_name}. Am I speaking with {self.call.patient.first_name}?",
            allow_interruptions=True,
        )

    async def _hangup(self) -> None:
        try:
            ctx = get_job_context()
            await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
        except Exception:
            logger.exception("failed to delete the room on hangup")

    # -- tools ------------------------------------------------------------

    @function_tool()
    async def check_availability(
        self,
        ctx: RunContext,
        specialty: str = "",
        earliest_in_days: int = 1,
    ) -> dict[str, Any]:
        """Look up open consultation slots at the clinic.

        Args:
            specialty: The medical specialty to book. Defaults to the one recommended for this patient.
            earliest_in_days: How many days from today the patient wants to start looking.
        """
        specialty = specialty or self.call.patient.recommended_specialty
        result = booking_service.get_availability(specialty, day_offset=max(1, earliest_in_days))
        logger.info("availability lookup: %s -> %d slots", specialty, result["count"])
        return result

    @function_tool()
    async def book_appointment(
        self,
        ctx: RunContext,
        date: str,
        time_of_day: str,
        slot_id: str = "",
        specialty: str = "",
        mode: str = "in_person",
        notes: str = "",
    ) -> dict[str, Any]:
        """Book the consultation. Call this only after the patient has agreed to a specific date and time.

        Args:
            date: Appointment date as YYYY-MM-DD.
            time_of_day: Appointment time as the patient would say it, for example "10:00 AM".
            slot_id: The slot_id from check_availability, when you have one.
            specialty: The specialty being booked.
            mode: Either "in_person" or "telehealth".
            notes: Anything the doctor should know, in one short sentence.
        """
        result = booking_service.book(
            patient_id=self.call.patient.patient_id,
            patient_name=self.call.patient.name,
            slot_id=slot_id or None,
            date=date,
            time_of_day=time_of_day,
            specialty=specialty or self.call.patient.recommended_specialty,
            mode=mode,
            notes=notes,
        )
        logger.info("booking attempt -> %s", result.get("status"))
        if result.get("status") == "confirmed":
            self.decision = {"decision": "booked", "details": result}
        return result

    @function_tool()
    async def record_patient_decision(
        self,
        ctx: RunContext,
        decision: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Record the patient's decision when they do not book right now.

        Args:
            decision: One of "declined", "callback_requested", "will_think_about_it", "not_interested".
            reason: What the patient said, in their words, in one short sentence.
        """
        self.decision = {"decision": decision, "reason": reason, "at": utc_now_iso()}
        logger.info("patient decision recorded: %s (%s)", decision, reason)
        return {"status": "recorded", "decision": decision}

    @function_tool()
    async def flag_for_urgent_followup(
        self,
        ctx: RunContext,
        symptoms: str,
    ) -> dict[str, Any]:
        """Flag the call for immediate clinician review after the patient reports urgent symptoms.

        Args:
            symptoms: What the patient described, verbatim if possible.
        """
        self.urgent_flag = {"symptoms": symptoms, "at": utc_now_iso()}
        logger.warning("URGENT follow-up flagged: %s", symptoms)
        return {
            "status": "flagged",
            "message": "A clinician has been alerted. Tell the patient someone will call them shortly.",
        }

    @function_tool()
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """Transfer the call to a human coordinator, after confirming the patient wants that."""
        transfer_to = self.call.transfer_to
        if not transfer_to:
            return "No human coordinator is available right now. Offer a callback instead."
        await ctx.session.generate_reply(
            instructions="Tell the patient you are transferring them to a colleague now."
        )
        try:
            job_ctx = get_job_context()
            await job_ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=self.participant.identity if self.participant else "",
                    transfer_to=f"tel:{transfer_to}",
                )
            )
            self.decision = {"decision": "transferred_to_human", "at": utc_now_iso()}
            return "transferred"
        except Exception as exc:  # noqa: BLE001
            logger.exception("transfer failed")
            return f"The transfer failed ({exc}). Apologise and offer a callback."

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext) -> None:
        """Call this when you hear a voicemail greeting rather than a live person."""
        logger.info("answering machine detected, hanging up")
        self.decision = {"decision": "voicemail", "at": utc_now_iso()}
        await self._hangup()

    @function_tool()
    async def end_call(self, ctx: RunContext) -> None:
        """End the call. Use this after saying goodbye, or when the patient asks to hang up."""
        logger.info("ending call")
        current = ctx.session.current_speech
        if current:
            await current.wait_for_playout()
        await self._hangup()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def prewarm(proc: JobProcess) -> None:
    """Warm everything that is slow or blocking, once per worker process."""
    proc.userdata["vad"] = silero.VAD.load()

    # litellm is a very heavy import, around two seconds. The online evaluators
    # reach for it during the shutdown callback, where importing it stalls the
    # event loop and eats into the shutdown budget. Pay that cost here instead.
    try:
        importlib.import_module("litellm")
    except Exception:  # noqa: BLE001
        logger.debug("litellm not importable at prewarm; judges will import it lazily")
    # Resolve and cache the voice here so the blocking HTTP checks never run on
    # the agent's event loop during a call.
    if elevenlabs is not None and (os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")):
        voice = os.getenv("ELEVEN_VOICE_ID", elevenlabs.DEFAULT_VOICE_ID)
        _usable_eleven_voice(voice)


# `AgentServer` is the current worker API in livekit-agents 1.8; the older
# `WorkerOptions(entrypoint_fnc=...)` still works but is no longer documented.
# `agent_name` is what makes dispatch explicit: the worker only takes jobs that
# `dispatch_call.py` sends it, instead of auto-joining every new room.
server = AgentServer(
    setup_fnc=prewarm,
    # The post-call pipeline runs in a shutdown callback: LLM analysis, then five
    # evaluators, some of which retry against a rate-limited provider. The 10s
    # default cuts that off midway and the Opik trace never closes.
    shutdown_process_timeout=float(os.getenv("SHUTDOWN_TIMEOUT_S", "180")),
)


def _resolve_provider(stage: str) -> str:
    """Pick a provider for one pipeline stage.

    Explicit env wins. Otherwise we prefer a provider whose key is present, and
    fall back to LiveKit Inference, which is served by LiveKit Cloud using the
    credentials this worker already has. That fallback means the agent runs with
    no model-provider key at all.
    """
    override = os.getenv(f"{stage.upper()}_PROVIDER", "").strip().lower()
    if override:
        return override

    # A provider counts only when its key is set AND its plugin actually imported.
    have = {
        "groq": bool(os.getenv("GROQ_API_KEY")) and groq is not None,
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "deepgram": bool(os.getenv("DEEPGRAM_API_KEY")) and deepgram is not None,
        "cartesia": bool(os.getenv("CARTESIA_API_KEY")) and cartesia is not None,
        "elevenlabs": bool(os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY"))
        and elevenlabs is not None,
    }

    if stage == "llm":
        # OpenAI first. Groq is faster per token, but the gpt-oss models it
        # serves are reasoning models: they think before answering, and on a
        # phone call that thinking is heard as silence. gpt-4o-mini starts
        # speaking immediately, which matters more here than raw throughput.
        for p in ("openai", "groq"):
            if have[p]:
                return p
    elif stage == "stt":
        # Deepgram first. Transcription is the latency-critical stage, and it is
        # what Deepgram is built for: streaming, 25ms endpointing, and keyterm
        # prompting on nova-3 so clinical words survive. Measured on a real call,
        # ElevenLabs transcription cost 2.7s per turn against 0.28s for speech
        # synthesis, so this is where the time was.
        for p in ("deepgram", "elevenlabs"):
            if have[p]:
                return p
        return "livekit"
    elif stage == "tts":
        for p in ("elevenlabs", "cartesia"):
            if have[p]:
                return p
        return "livekit"
    return "livekit"


_ELEVEN_VOICE_CACHE: dict[str, str] = {}


def _usable_eleven_voice(voice_id: str) -> str:
    """Fall back to a premade voice when the configured one is not accessible.

    ElevenLabs returns 402 for shared library voices on a free account, and that
    failure would otherwise land mid-call as a dead agent.

    The result is cached per worker process and resolved during prewarm, because
    these are blocking HTTP calls. Running them inside the job would stall the
    agent's event loop and delay audio, which LiveKit rightly flags.
    """
    if voice_id in _ELEVEN_VOICE_CACHE:
        return _ELEVEN_VOICE_CACHE[voice_id]
    resolved = _resolve_eleven_voice(voice_id)
    _ELEVEN_VOICE_CACHE[voice_id] = resolved
    return resolved


def _resolve_eleven_voice(voice_id: str) -> str:
    fallback = "EXAVITQu4vr4xnSDxMaL"  # Sarah, premade, available on every tier
    key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
    if not voice_id or not key or voice_id == fallback:
        return voice_id or fallback

    def _get(path: str) -> dict[str, Any]:
        import urllib.request

        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/{path}", headers={"xi-api-key": key}
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            return json.loads(resp.read().decode())

    try:
        # Fetching the voice always succeeds; it is synthesis that is refused, so
        # check the two things that actually decide it: the plan and the category.
        tier = str(_get("user/subscription").get("tier", "")).lower()
        category = str(_get(f"voices/{voice_id}").get("category", "")).lower()
    except Exception:
        # A startup hiccup should not override an explicit choice.
        return voice_id

    if tier in ("free", "") and category != "premade":
        logger.warning(
            "ElevenLabs voice %s is a '%s' voice and this account is on the '%s' plan, "
            "which can only synthesize 'premade' voices via the API. Falling back to %s. "
            "Upgrade the plan to use the configured voice.",
            voice_id, category or "unknown", tier or "unknown", fallback,
        )
        return fallback
    return voice_id


def _speech_keyterms(patient: PatientRecord) -> list[str]:
    """Vocabulary hints for the transcriber, drawn from this patient's record.

    Clinical terms are exactly what general speech models get wrong, and a
    mistranscribed biomarker name is the one error this agent cannot tolerate.
    """
    terms = {patient.recommended_specialty, patient.care_program, patient.first_name}
    for b in patient.biomarkers:
        terms.add(b.label)
        terms.update(w for w in b.label.split() if len(w) > 3)
    terms.update(
        ["HbA1c", "biomarker", "endocrinologist", "telehealth", "cholesterol", "glucose"]
    )

    # ElevenLabs realtime transcription rejects any keyterm over 20 characters,
    # and rejects the whole connection rather than the offending term: a live
    # call died on "Fasting blood glucose" at 21 characters. Long labels are
    # dropped in favour of their individual words, which carry the same hint.
    cleaned: set[str] = set()
    for term in terms:
        term = (term or "").strip()
        if not term:
            continue
        if len(term) <= MAX_KEYTERM_CHARS:
            cleaned.add(term)
            continue
        cleaned.update(
            w for w in term.split() if 3 < len(w) <= MAX_KEYTERM_CHARS
        )
    return sorted(cleaned)


def _build_realtime_session(vad: Any) -> tuple[AgentSession, dict[str, str]]:
    """Speech-to-speech. One model, one hop, no cascade."""
    realtime = openai.realtime.RealtimeModel(
        model=REALTIME_MODEL,
        voice=REALTIME_VOICE,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    logger.info("pipeline: realtime speech-to-speech (%s, voice=%s)", REALTIME_MODEL, REALTIME_VOICE)
    return (
        AgentSession(llm=realtime, vad=vad),
        {
            "mode": "realtime",
            "llm_provider": "openai",
            "llm_model": REALTIME_MODEL,
            "voice": REALTIME_VOICE,
        },
    )


def _build_session(vad: Any, keyterms: list[str] | None = None) -> tuple[AgentSession, dict[str, str]]:
    """Assemble the STT -> LLM -> TTS pipeline from whatever keys are available.

    Returns the session and a description of which providers were chosen, which
    gets recorded on the Opik trace so a bad call can be tied to a model.
    """
    from livekit.agents import inference

    stt_provider = _resolve_provider("stt")
    llm_provider = _resolve_provider("llm")
    tts_provider = _resolve_provider("tts")

    # --- speech to text ---
    keyterms = keyterms or []
    if stt_provider == "elevenlabs":
        # `scribe_v1` is the default because it works on a free ElevenLabs account.
        # Paid accounts should set ELEVEN_STT_MODEL=scribe_v2_realtime, which
        # streams and cuts turn latency noticeably on a phone call.
        # Patients answer the phone in whatever language is natural to them. A
        # real call to an Indian number came back with "हाँ जी" for yes, so the
        # secondary languages are configurable rather than pinned to English.
        secondary = [
            code.strip()
            for code in os.getenv("ELEVEN_STT_SECONDARY_LANGUAGES", "hi").split(",")
            if code.strip()
        ]
        # scribe_v2_realtime streams; scribe_v1 does not. On a real call the
        # non-streaming model cost 3.9s of transcription delay per turn, which
        # was the whole of the perceived lag: text-to-speech was only 0.28s.
        # scribe_v1 remains the fallback for free accounts.
        stt: Any = elevenlabs.STT(
            model=os.getenv("ELEVEN_STT_MODEL", "scribe_v2_realtime"),
            language_code=os.getenv("ELEVEN_STT_LANGUAGE", "en"),
            secondary_languages=secondary or NOT_GIVEN,
            include_language_detection=bool(secondary),
            # Biases transcription toward this patient's own terminology, so
            # "HbA1c" and "endocrinologist" come back spelled correctly.
            keyterms=keyterms or NOT_GIVEN,
        )
    elif stt_provider == "deepgram":
        stt = deepgram.STT(model=os.getenv("DEEPGRAM_MODEL", "nova-3"))
    elif stt_provider == "groq":
        stt = groq.STT(model=os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"))
    elif stt_provider == "openai":
        stt = openai.STT(model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe"))
    else:
        # `nova-3-medical` is worth it here: the patient will say words like
        # "HbA1c" and "endocrinologist" that general models mistranscribe.
        stt = inference.STT(model=os.getenv("LIVEKIT_STT_MODEL", "deepgram/nova-3-medical"))

    # --- language model ---
    if llm_provider == "groq":
        llm_model = LLM_MODEL
        llm_kwargs: dict[str, Any] = {"model": llm_model, "temperature": 0.4}
        if "gpt-oss" in llm_model and GROQ_REASONING_EFFORT:
            llm_kwargs["reasoning_effort"] = GROQ_REASONING_EFFORT
        llm: Any = groq.LLM(**llm_kwargs)
    elif llm_provider == "openai":
        llm_model = LLM_MODEL
        llm = openai.LLM(model=llm_model, temperature=0.4)
    else:
        llm_model = os.getenv("LIVEKIT_LLM_MODEL", "openai/gpt-4o-mini")
        llm = inference.LLM(model=llm_model)

    # --- text to speech ---
    if tts_provider == "elevenlabs":
        # Slower than default on purpose. This agent delivers worrying lab
        # results to people who may be older or anxious, and default TTS pace
        # is noticeably too quick to absorb a number like "seven point eight".
        tts: Any = elevenlabs.TTS(
            voice_id=_usable_eleven_voice(
                os.getenv("ELEVEN_VOICE_ID", elevenlabs.DEFAULT_VOICE_ID)
            ),
            model=os.getenv("ELEVEN_TTS_MODEL", "eleven_turbo_v2_5"),
            language=os.getenv("ELEVEN_TTS_LANGUAGE", "en"),
            voice_settings=elevenlabs.VoiceSettings(
                stability=float(os.getenv("ELEVEN_STABILITY", "0.6")),
                similarity_boost=float(os.getenv("ELEVEN_SIMILARITY", "0.75")),
                speed=float(os.getenv("ELEVEN_SPEED", "0.85")),
                use_speaker_boost=True,
            ),
        )
    elif tts_provider == "cartesia":
        tts = cartesia.TTS()
    elif tts_provider == "groq":
        tts = groq.TTS(
            model=os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english"),
            voice=os.getenv("GROQ_TTS_VOICE", "autumn"),
        )
    elif tts_provider == "openai":
        tts = openai.TTS(model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"), voice=TTS_VOICE)
    else:
        tts = inference.TTS(
            model=os.getenv("LIVEKIT_TTS_MODEL", "cartesia/sonic-2"),
            voice=os.getenv("LIVEKIT_TTS_VOICE", "f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        )

    logger.info(
        "pipeline: stt=%s | llm=%s (%s) | tts=%s",
        stt_provider, llm_provider, llm_model, tts_provider,
    )

    kwargs: dict[str, Any] = {"vad": vad, "stt": stt, "tts": tts, "llm": llm}

    # The turn detector is a local model that needs `python agent.py download-files`.
    # It measurably improves phone-call turn taking, but the agent runs without it.
    # Turn handling is left at the SDK defaults on purpose.
    #
    # An earlier version overrode endpointing and switched on preemptive
    # generation to shave latency. It broke the conversation outright: the
    # greeting played, the patient's reply was transcribed, and the agent then
    # sat silent because the turn never completed. The worker log showed zero
    # requests to the language model for the whole session.
    #
    # Endpointing is tunable through the environment for anyone who wants to
    # experiment, but nothing is overridden unless it is set explicitly, so the
    # default path is the one that is known to work.
    turn_handling: dict[str, Any] = {}
    if EnglishModel is not None:
        turn_handling["turn_detection"] = EnglishModel()
    else:
        logger.info("turn-detector model unavailable, falling back to VAD turn detection")

    endpointing: dict[str, Any] = {}
    if os.getenv("ENDPOINTING_MIN_DELAY"):
        endpointing["min_delay"] = float(os.environ["ENDPOINTING_MIN_DELAY"])
    if os.getenv("ENDPOINTING_MAX_DELAY"):
        endpointing["max_delay"] = float(os.environ["ENDPOINTING_MAX_DELAY"])
    if endpointing:
        logger.warning("overriding endpointing defaults with %s", endpointing)
        turn_handling["endpointing"] = endpointing

    if turn_handling:
        kwargs["turn_handling"] = turn_handling

    pipeline_info = {
        "stt_provider": stt_provider,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "tts_provider": tts_provider,
        "turn_detection": "english_model" if EnglishModel is not None else "vad",
    }
    return AgentSession(**kwargs), pipeline_info


def _resolve_call_context(ctx: JobContext) -> CallContext:
    """Rebuild the call from job metadata, or fall back to a local demo patient."""
    metadata = (ctx.job.metadata or "").strip()
    if metadata:
        try:
            return CallContext.from_metadata(metadata)
        except Exception:
            logger.exception("could not parse job metadata, falling back to the demo patient")

    patient: PatientRecord = load_patients()[0]
    logger.warning(
        "no job metadata; running in simulation with demo patient %s", patient.patient_id
    )
    return CallContext(
        patient=patient,
        call_id=f"sim-{uuid.uuid4().hex[:8]}",
        simulate=True,
    )


def _persist_recording(session_report_path: Path | None, call_id: str) -> str | None:
    """Copy the recording out of the job's temp directory before it is cleaned up."""
    if not session_report_path or not Path(session_report_path).exists():
        return None
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RECORDINGS_DIR / f"{call_id}{Path(session_report_path).suffix or '.ogg'}"
    try:
        shutil.copy2(session_report_path, dest)
        logger.info("call recording saved to %s", dest)
        return str(dest)
    except OSError:
        logger.exception("could not copy the call recording")
        return None


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    call = _resolve_call_context(ctx)
    patient = call.patient
    ctx.log_context_fields = {"call_id": call.call_id, "patient_id": patient.patient_id}

    logger.info(
        "starting call %s to %s (%s), simulate=%s",
        call.call_id, patient.name, patient.phone_number, call.simulate,
    )

    await ctx.connect()

    agent = HealthcareOutboundAgent(call)
    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()
    if VOICE_MODE == "realtime" and os.getenv("OPENAI_API_KEY"):
        session, pipeline_info = _build_realtime_session(vad)
    else:
        session, pipeline_info = _build_session(vad, keyterms=_speech_keyterms(patient))

    # --- Opik: line 1 of 3 -------------------------------------------------
    observer = await OpikCallObserver.astart(
        call_id=call.call_id,
        call_name=f"outbound_call::{patient.name}::{call.call_id}",
        variables={
            "call_id": call.call_id,
            "patient_id": patient.patient_id,
            "patient_name": patient.name,
            "phone_number": patient.phone_number,
            "clinic": call.clinic_name,
            "agent_persona": call.agent_display_name,
            "recommended_specialty": patient.recommended_specialty,
            "care_program": patient.care_program,
            "urgency": patient.urgency,
            "room": ctx.room.name,
            "job_id": ctx.job.id,
            "simulated_telephony": call.simulate,
            **pipeline_info,
        },
        grounding=patient.to_dict(),
        tags=["livekit", "voice", "outbound", "healthcare", f"program:{patient.care_program}"],
    )

    # --- Opik: line 2 of 3 -------------------------------------------------
    observer.attach(session)

    call_started = time.time()
    dial_result: dict[str, Any] = {"attempted": False}

    # --- Opik: line 3 of 3 -------------------------------------------------
    async def on_shutdown(reason: str = "") -> None:
        await _finalize_call(
            ctx=ctx,
            call=call,
            agent=agent,
            session=session,
            observer=observer,
            dial_result=dial_result,
            call_started=call_started,
            shutdown_reason=reason,
        )

    ctx.add_shutdown_callback(on_shutdown)

    # Start the session before dialing so nothing the patient says on pickup is lost.
    session_task = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_options=_room_options(),
            # Audio is always recorded locally: that file is what gets attached
            # to the Opik trace. The other three push the session to LiveKit
            # Cloud's own observability view, which is useful as a second place
            # to inspect a call. Turn them off with LIVEKIT_CLOUD_RECORDING=false
            # if you would rather everything lived only in Opik.
            record={
                "audio": True,
                "traces": LIVEKIT_CLOUD_RECORDING,
                "logs": LIVEKIT_CLOUD_RECORDING,
                "transcript": LIVEKIT_CLOUD_RECORDING,
            },
        )
    )

    if call.simulate or not OUTBOUND_TRUNK_ID:
        if not call.simulate:
            logger.warning("SIP_OUTBOUND_TRUNK_ID is not set; running without telephony")
        dial_result = {"attempted": False, "mode": "simulated", "status": "in_room_only"}
        await session_task
        logger.info("simulated call ready in room %s; join it to talk to the agent", ctx.room.name)
        # Same rule as the phone path: do not speak into an empty room. Wait for
        # a human to actually join before the agent opens its mouth.
        participant = await ctx.wait_for_participant()
        agent.set_participant(participant)
        dial_result["status"] = "answered"
        logger.info("participant joined: %s", participant.identity)
        agent.greet()
        return

    dial_result = {"attempted": True, "mode": "sip", "phone_number": patient.phone_number}
    try:
        sip_participant = await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=OUTBOUND_TRUNK_ID,
                sip_call_to=patient.phone_number,
                participant_identity=patient.phone_number,
                participant_name=patient.name,
                krisp_enabled=True,
                wait_until_answered=True,
            )
        )
        dial_result.update(
            {
                "status": "answered",
                "participant_id": getattr(sip_participant, "participant_id", None),
                "answered_after_s": round(time.time() - call_started, 2),
            }
        )
        await session_task
        participant = await ctx.wait_for_participant(identity=patient.phone_number)
        agent.set_participant(participant)
        logger.info("patient answered: %s", participant.identity)
        # Only now is there an ear on the other end. Greeting any earlier plays
        # the opening line into a ringing line and the patient misses it.
        agent.greet()
    except api.TwirpError as exc:
        sip_code = exc.metadata.get("sip_status_code") if exc.metadata else None
        sip_status = exc.metadata.get("sip_status") if exc.metadata else None
        dial_result.update(
            {
                "status": "failed",
                "error": exc.message,
                "sip_status_code": sip_code,
                "sip_status": sip_status,
            }
        )
        logger.error("SIP dial failed: %s (%s %s)", exc.message, sip_code, sip_status)
        ctx.shutdown(reason="sip_dial_failed")
    except Exception as exc:  # noqa: BLE001
        dial_result.update({"status": "failed", "error": str(exc)})
        logger.exception("unexpected dialing failure")
        ctx.shutdown(reason="dial_error")


def _room_options() -> RoomOptions:
    """Krisp telephony noise cancellation, when the plugin is installed."""
    if noise_cancellation is None:
        logger.info("noise-cancellation plugin unavailable, continuing without it")
        return RoomOptions()
    return RoomOptions(
        audio_input=AudioInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
        )
    )


async def _finalize_call(
    *,
    ctx: JobContext,
    call: CallContext,
    agent: HealthcareOutboundAgent,
    session: AgentSession,
    observer: OpikCallObserver,
    dial_result: dict[str, Any],
    call_started: float,
    shutdown_reason: str,
) -> None:
    """Post-call: recording, transcript, analysis, Opik. Never raises."""
    logger.info("call %s ended (%s), running post-call pipeline", call.call_id, shutdown_reason)

    transcript = observer.transcript
    tool_calls = observer.tool_calls

    # The session history is the SDK's own record; prefer the live capture but
    # fall back to it if the event stream gave us nothing.
    if not transcript:
        try:
            history = session.history.to_dict()
            for item in history.get("items", []):
                if item.get("type") == "message" and item.get("role") in ("user", "assistant"):
                    content = item.get("content")
                    text = " ".join(c for c in content if isinstance(c, str)) if isinstance(content, list) else str(content)
                    if text.strip():
                        transcript.append({"role": item["role"], "content": text})
        except Exception:
            logger.exception("could not read session history")

    # Recording: copy it out of the job temp dir before cleanup removes it.
    audio_path: str | None = None
    try:
        report = ctx.make_session_report(session)
        audio_path = _persist_recording(report.audio_recording_path, call.call_id)
    except Exception:
        logger.warning("session report unavailable, looking for the raw recording file")
        try:
            candidate = Path(ctx.session_directory) / "audio.ogg"
            audio_path = _persist_recording(candidate, call.call_id)
        except Exception:
            logger.exception("no call recording could be recovered")

    telemetry = {
        "duration_s": round(time.time() - call_started, 2),
        "shutdown_reason": shutdown_reason,
        "dial": dial_result,
        "room": ctx.room.name,
        "job_id": ctx.job.id,
        "agent_decision": agent.decision,
        "urgent_flag": agent.urgent_flag,
        "simulated_telephony": call.simulate,
    }

    analysis: dict[str, Any] = {}
    try:
        analysis = await analyze_call(
            patient=call.patient.to_dict(),
            transcript=transcript,
            tool_calls=tool_calls,
            telemetry=telemetry,
        )
        if agent.urgent_flag:
            analysis["escalation_needed"] = True
        logger.info(
            "post-call analysis: outcome=%s booked=%s",
            analysis.get("outcome"), analysis.get("appointment_booked"),
        )
    except Exception:
        logger.exception("post-call analysis failed")
        analysis = {"outcome": "analysis_failed", "appointment_booked": False}

    # Local artifact, so the run is inspectable without opening Opik.
    try:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        artifact = {
            "call_id": call.call_id,
            "patient": call.patient.to_dict(),
            "telemetry": telemetry,
            "transcript": transcript,
            "tool_calls": tool_calls,
            "analysis": analysis,
            "recording_path": audio_path,
        }
        out = TRANSCRIPTS_DIR / f"{call.call_id}.json"
        out.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
        logger.info("call artifact written to %s", out)
    except Exception:
        logger.exception("could not write the local call artifact")

    result = await observer.finalize(
        analysis=analysis,
        audio_path=audio_path,
        audio_reference=f"livekit://{ctx.room.name}/{call.call_id}",
        telemetry=telemetry,
        transcript=transcript,
        tool_calls=tool_calls,
    )
    logger.info("opik finalize: %s", result)


if __name__ == "__main__":
    cli.run_app(server)
