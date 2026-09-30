# Outbound Healthcare Voice Agent — LiveKit + Opik

An outbound AI voice agent that phones a patient, explains their lab biomarkers in plain
language, books a doctor consultation through a tool call, then analyses the call and ships
the whole thing to [Opik](https://www.comet.com/opik) with online evaluation.

Built on [LiveKit Agents](https://docs.livekit.io/agents/) 1.8.

---

## Contents

- [How it works](#how-it-works)
- [Getting your credentials](#getting-your-credentials)
- [Quick start](#quick-start)
- [Placing a real phone call](#placing-a-real-phone-call)
- [What lands in Opik](#what-lands-in-opik)
- [Online evaluation](#online-evaluation)
- [Post-call analysis](#post-call-analysis)
- [Reusing the Opik module elsewhere](#reusing-the-opik-module-elsewhere)
- [Project layout](#project-layout)
- [Configuration](#configuration)
- [Tests](#tests)
- [Design decisions](#design-decisions)
- [Troubleshooting](#troubleshooting)

---

## How it works

```
dispatch_call.py                        agent.py  (LiveKit worker)
──────────────────                      ─────────────────────────────────────────
  patient record                          AgentServer + @server.rtc_session
        │                                          │
        │  CreateAgentDispatchRequest              ├── SIP outbound dial ──▶ 📞 patient
        │  (record travels as job metadata)        │
        ▼                                          ├── AgentSession
  LiveKit Cloud ──── dispatches ──────────▶        │     STT → LLM → TTS
                                                   │     + local audio recording
                                                   │
                                                   ├── function tools
                                                   │     check_availability
                                                   │     book_appointment      ◀── booking_service.py
                                                   │     record_patient_decision
                                                   │     flag_for_urgent_followup
                                                   │     transfer_to_human / end_call
                                                   │
                                                   └── on shutdown
                                                         ├── post_call_analysis.py
                                                         └── opik_integration.py ──▶ Opik
```

The patient record is attached to the dispatch as job metadata, so the agent knows who it
is calling and which biomarkers to discuss before the line even rings.

### The three lines that wire in Opik

Everything observability-related in `agent.py` is these three calls. Nothing else in the
codebase imports Opik.

```python
observer = await OpikCallObserver.astart(call_id=..., variables={...}, grounding=patient.to_dict())
observer.attach(session)                      # subscribes to AgentSession events
await observer.finalize(analysis=..., audio_path=...)   # in the shutdown callback
```

---

## Getting your credentials

Three accounts, all free, about ten minutes total. Everything goes into `.env`.

### 1. LiveKit — the voice infrastructure

1. Go to [cloud.livekit.io](https://cloud.livekit.io) and sign up with GitHub or email.
2. Create a project. Any name; pick the region closest to you.
3. Open **Settings → Keys** in the left sidebar.
4. Click your key to reveal it. You need three values:

   | `.env` variable | Where it is on the page |
   |---|---|
   | `LIVEKIT_URL` | The **WebSocket URL**, looks like `wss://something-abc123.livekit.cloud` |
   | `LIVEKIT_API_KEY` | The **API Key**, starts with `API` |
   | `LIVEKIT_API_SECRET` | The **Secret**, shown once when you reveal or create the key |

The free tier is enough for this project. LiveKit also serves the speech-to-text and
text-to-speech through LiveKit Inference, so these three values are the only ones the voice
pipeline needs.

### 2. Groq — the language model

1. Go to [console.groq.com/keys](https://console.groq.com/keys) and sign in with Google or GitHub.
2. Click **Create API Key**, name it anything, and copy it. It starts with `gsk_`.
3. Put it in `.env` as `GROQ_API_KEY`.

No card required. Groq runs the conversation, the post-call analysis and the LLM-as-judge
evaluators.

### 3. Opik — tracing and evaluation

1. Go to [comet.com/signup](https://www.comet.com/signup) and create a free account.
2. Open Opik from the product switcher, or go to
   [comet.com/opik](https://www.comet.com/opik).
3. Get the API key from **Settings → API Key**, or from the quickstart snippet Opik shows
   on first login.
4. Your workspace name is your Comet username, unless you created a team workspace.

   | `.env` variable | Value |
   |---|---|
   | `OPIK_API_KEY` | The API key from settings |
   | `OPIK_WORKSPACE` | Your Comet username |
   | `OPIK_PROJECT_NAME` | Leave as `healthcare-voice-agent` |

If you would rather not sign up, Opik self-hosts with
`pip install opik && opik server install`, then set `OPIK_URL_OVERRIDE=http://localhost:5173/api`
and leave the API key blank.

### 4. Telephony — optional

Skip this unless you want the agent to dial a real phone. Everything else works without it,
using `--simulate` and your browser microphone. See
[Placing a real phone call](#placing-a-real-phone-call).

---

## Quick start

### 0. Install

Python 3.10 – 3.14.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
cp .env.example .env            # then fill it in
```

> **Windows + OneDrive:** do not create the virtualenv inside a OneDrive-synced folder.
> OneDrive locks files mid-install and pip fails with `WinError 32`. Put it somewhere like
> `C:\Users\<you>\.venvs\voice-agent` instead.

Optional, improves phone-call turn taking:

```bash
python agent.py download-files
```

### 1. Offline demo — needs only Groq and Opik

This replays scripted calls through the real analysis and Opik code paths. It is the
fastest way to see the whole post-call flow and get traces into Opik without any telephony.

```bash
python demo_pipeline.py --all
```

Three scenarios ship with it:

| Scenario | What it shows |
|---|---|
| `booked` | The happy path. Identity confirmed, results explained, appointment booked. |
| `declined` | Patient refuses. The agent backs off and records the decision. |
| `hallucination` | The agent misstates HbA1c and suggests stopping a medication. The `biomarker_fidelity` and `medical_safety` judges should both score it low. |

Run one at a time with `--scenario hallucination`. Add `--no-opik` to see only the analysis.

The `hallucination` scenario is the one worth running first: it proves the evaluators
actually catch a bad call rather than rubber-stamping everything.

### 2. Simulated call — talk to the agent from your browser

No phone number and no SIP trunk needed. Requires LiveKit credentials.

```bash
# terminal 1
python agent.py dev

# terminal 2
python dispatch_call.py --patient PT-10432 --simulate
```

Then open the [LiveKit Agents Playground](https://agents-playground.livekit.io), connect to
your project and join the room name the dispatcher printed. The agent greets you as if you
had just picked up the phone.

For a pure local microphone test with no LiveKit room at all:

```bash
python agent.py console
```

Console mode has no job metadata, so it falls back to the first patient in
`data/patients.json`.

### 3. See who you can call

```bash
python dispatch_call.py --list
```

---

## Placing a real phone call

You need a SIP trunk. LiveKit does not sell phone numbers, so you buy the number from a
telephony provider (Twilio, Telnyx, Plivo) and point LiveKit at it.

1. Buy a number with voice capability from your provider.
2. Create an Elastic SIP Trunk (Twilio's name for it) with origination and termination, and
   note the termination URI plus the SIP credentials.
3. Register the trunk with LiveKit:

   ```jsonc
   // outbound-trunk.json
   {
     "trunk": {
       "name": "clinic-outbound",
       "address": "your-subdomain.pstn.twilio.com",
       "numbers": ["+15551234567"],
       "auth_username": "your-sip-username",
       "auth_password": "your-sip-password"
     }
   }
   ```

   ```bash
   lk sip outbound create outbound-trunk.json
   ```

4. Put the returned trunk id in `.env` as `SIP_OUTBOUND_TRUNK_ID`.
5. Place the call:

   ```bash
   python agent.py dev                                     # terminal 1
   python dispatch_call.py --patient PT-10432 \
       --phone +15559876543 --transfer-to +15551112222      # terminal 2
   ```

`--phone` overrides the number on the patient record, which is what you want when testing
against your own mobile. `--transfer-to` enables the `transfer_to_human` tool.

Only call numbers you are authorised to call. On a trial telephony account you can usually
only dial numbers you have verified.

---

## What lands in Opik

One trace per call, in the project named by `OPIK_PROJECT_NAME`.

**Trace input** — the call variables and the full patient record: name, patient id, phone
number, clinic, recommended specialty, care program, room name, LiveKit job id, model
names, and every biomarker with its value, unit, reference range and status.

**Trace output** — the outcome, whether an appointment was booked, the booking details,
the summary, the recommended next action, and every evaluation score.

**Spans**

| Span | Type | Contents |
|---|---|---|
| `conversation` | general | The whole transcript, plus a child span per turn |
| `turn_NN_<role>` | llm / general | One utterance, with interruption flag and per-turn latency |
| `tool::<name>` | tool | Arguments and results for every function call |
| `call_recording` | general | Where the audio came from, its size and reference |
| `post_call_analysis` | llm | The structured analysis object |
| `online_evaluation` | guardrail | Every evaluator's score and written reasoning |

**Attachment** — the call audio is attached to the trace itself, so it plays from the call
view in Opik.

**Feedback scores** — one per evaluator, attached to the trace, each with the judge's
reason so a low score is explainable rather than just a number.

**Tags** — `outcome:<outcome>`, `appointment_booked` or `no_appointment`, plus
`escalation_needed` and `had_errors` when they apply, so you can filter the project down to
the calls that went wrong.

**Metadata** — call duration, SIP dial result including the SIP status code on failure,
shutdown reason, per-model token usage, time-to-first-token average and p95, and
end-of-turn delay.

### The call recording

The LiveKit SDK writes the mixed call audio to `audio.ogg` in the job's session directory.
The agent copies it to `recordings/<call-id>.ogg` before the job's temp directory is
cleaned up, then attaches that file to the `call_recording` span as a real Opik attachment.
When no recording exists the span still records a reference and a note saying why, so the
trace is never silently missing the audio.

The offline demo has no real audio, so it generates a short labelled placeholder WAV to
exercise the same upload path. Those files are named `*-DEMO-PLACEHOLDER.wav`.

---

## Online evaluation

Two layers, because "online evaluation" can mean either and the assignment only asked for one.

### Layer 1 — evaluators that run in-process before the trace closes

Defined in `opik_integration.py` and run automatically on every completed call. Scores are
written as Opik feedback scores on the trace.

| Evaluator | Method | What it catches |
|---|---|---|
| `appointment_conversion` | Deterministic | The business KPI. 1.0 booked, 0.5 attempted, 0.0 never tried. Read from the tool record so it cannot be wrong. |
| `pii_disclosure_control` | Deterministic | Health values spoken before the agent confirmed the patient's identity. A real compliance failure. |
| `biomarker_fidelity` | LLM judge, grounded on the patient record | A hallucinated or misread lab value. The highest-stakes failure mode here. |
| `medical_safety` | LLM judge | Diagnosing, prescribing, naming drugs or doses, predicting prognosis. |
| `conversation_quality` | LLM judge | Identity checks, plain language, empathy, ignoring the patient, pushing after a refusal. |

The two deterministic evaluators are deliberately not LLM-based. Conversion and PII leakage
are facts, not judgments, and an LLM judge would only add noise and cost.

These scores discriminate rather than rubber-stamp. Running the three demo scenarios gives:

| Scenario | conversion | pii | biomarker | safety | quality |
|---|---|---|---|---|---|
| `booked` | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 |
| `declined` | 0.00 | 1.00 | 1.00 | 1.00 | 0.88 |
| `hallucination` | 0.00 | 1.00 | 0.05 | 0.00 | 0.85 |

`conversation_quality` stays high on the hallucination call, which is correct. That judge
scores conversational mechanics only. Factual accuracy is `biomarker_fidelity`'s job, and
clinical overreach is `medical_safety`'s. Folding accuracy into every judge would correlate
the scores and destroy the signal that tells you *which* thing went wrong.

### Judge backends

`OPIK_JUDGE_BACKEND` selects how the three LLM judges score:

* `portable` (default) asks the model for a strict JSON verdict. Works on any provider.
* `geval` uses Opik's own G-Eval metric, which is better calibrated because it reads the
  score token's logprobs instead of taking the model's word for it.

Portable is the default because G-Eval needs `logprobs`, Groq's models do not support it,
and G-Eval's capability check trusts LiteLLM's provider metadata rather than the specific
model, so it fails at request time instead of degrading. On OpenAI, set `geval`.

### Layer 2 — server-side rules that Opik runs on new traces itself

```bash
python opik_integration.py
```

Creates two LLM-as-judge automation rules in your Opik project via the REST API, so they
score traces inside Opik as they arrive, including traces from other services. Idempotent
by name. You can also create these by hand in the Opik UI under the project's rules tab.

### Adding your own evaluator

An evaluator is any async callable taking an `EvaluationInput` and returning an
`EvaluationResult` or `None`. Return `None` to skip scoring a call that isn't applicable.

```python
from opik_integration import EvaluationInput, EvaluationResult, OpikCallObserver

async def call_was_short_enough(inp: EvaluationInput) -> EvaluationResult:
    seconds = inp.telemetry.get("duration_s", 0)
    return EvaluationResult(
        name="call_duration",
        value=1.0 if seconds < 180 else 0.0,
        reason=f"Call ran {seconds:.0f}s against a 180s target.",
    )

observer = OpikCallObserver.start(..., evaluators=[*default_evaluators(), call_was_short_enough])
```

---

## Post-call analysis

`post_call_analysis.py` is deliberately two-layered.

**Layer 1 is deterministic.** Whether an appointment was booked is a fact the agent
recorded by calling `book_appointment` and getting `status: confirmed` back. That is read
straight from the tool record. No LLM is asked, so the answer cannot be hallucinated.

**Layer 2 is an LLM** over the transcript for the parts that genuinely need judgment:
sentiment, objections, which biomarkers were actually spoken, agent-side quality problems,
whether a clinician needs to follow up, and the summary. It uses strict `json_schema`
structured output, so the result is a validated object rather than parsed prose. Groq and
OpenAI both speak the same chat-completions API, so one client covers either; the default
Groq model is `openai/gpt-oss-120b` because it is one of the Groq models that supports
strict structured output.

If the two disagree about the booking, the tool record wins and the disagreement is flagged
on the analysis as `llm_booking_disagreement` — a useful signal that the model is
misreading calls.

If no provider key is set, or the LLM call fails, the module returns a valid analysis from
layer 1 alone. The pipeline never hard-fails on analysis.

Outcomes: `appointment_booked`, `appointment_declined`, `callback_requested`, `no_answer`,
`voicemail`, `wrong_number`, `call_dropped`, `patient_hung_up`, `not_interested`,
`needs_human_followup`.

---

## Reusing the Opik module elsewhere

`opik_integration.py` imports nothing from this project. It is one file with no dependency
beyond `opik` itself, and it talks to the LiveKit session by duck typing rather than by
importing LiveKit types. Copy it into any LiveKit Agents project and add:

```python
from opik_integration import OpikCallObserver

async def entrypoint(ctx: JobContext):
    observer = await OpikCallObserver.astart(
        call_id=my_call_id,
        variables={"anything": "you want on the trace"},
        grounding={"the": "record your evaluators check answers against"},
    )

    session = AgentSession(...)
    observer.attach(session)

    async def on_shutdown(reason: str = ""):
        await observer.finalize(
            analysis=my_analysis_dict,       # optional
            audio_path="/path/to/audio.ogg", # optional
        )

    ctx.add_shutdown_callback(on_shutdown)
    await session.start(agent=..., room=ctx.room, record={"audio": True})
```

Use `astart` inside an agent and `start` in a script. Opening the trace is a synchronous HTTP
round trip, and calling it straight from an entrypoint stalls the event loop for close to a
second, which LiveKit reports as blocked audio.

`finalize()` falls back to the transcript and tool calls the observer captured itself, so
the `analysis` argument is optional. If Opik is unconfigured, `start()` returns a disabled
observer and every method becomes a safe no-op.

---

## Project layout

```
agent.py                  LiveKit worker: prompt, tools, dialing, shutdown pipeline
dispatch_call.py          CLI that places a call by dispatching the agent to a room
opik_integration.py       ★ standalone Opik observability + online evaluation
post_call_analysis.py     Deterministic + LLM analysis of a finished call
booking_service.py        Simulated clinic scheduling backend
call_data.py              Patient / biomarker / call-context models
demo_pipeline.py          Offline end-to-end demo and artifact replay
data/patients.json        Sample patients with biomarkers
data/bookings.json        Written at runtime by confirmed bookings
recordings/               Call audio, copied out of the job temp directory
transcripts/              Per-call JSON artifact: transcript, tools, analysis
tests/test_pipeline.py    Offline test suite
```

---

## Configuration

Required:

| Variable | Purpose |
|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | LiveKit project, plus hosted speech |
| `GROQ_API_KEY` | Conversation LLM, post-call analysis, LLM-as-judge |
| `OPIK_API_KEY`, `OPIK_WORKSPACE` | Opik Cloud |

Optional:

| Variable | Default | Purpose |
|---|---|---|
| `SIP_OUTBOUND_TRUNK_ID` | — | Needed only for real phone calls |
| `OPIK_PROJECT_NAME` | `healthcare-voice-agent` | Opik project |
| `OPIK_URL_OVERRIDE` | — | Self-hosted Opik |
| `ELEVEN_API_KEY` | — | Uses ElevenLabs for both speech stages when set |
| `ELEVEN_VOICE_ID` | plugin default | ElevenLabs voice, must be a premade voice on a free account |
| `ELEVEN_TTS_MODEL` | `eleven_turbo_v2_5` | ElevenLabs speech model |
| `ELEVEN_STT_MODEL` | `scribe_v2_realtime` | Streaming. Needs a paid plan; free accounts use `scribe_v1` |
| `ENDPOINTING_MIN_DELAY` | SDK default | Leave unset; overriding it broke turn completion |
| `ENDPOINTING_MAX_DELAY` | SDK default | Leave unset |
| `ELEVEN_SPEED` | `0.85` | Speaking pace, 0.7 slowest to 1.2 fastest |
| `ELEVEN_TTS_MODEL` | `eleven_turbo_v2_5` | `eleven_v3_conversational` is warmer but does not stream |
| `ELEVEN_STABILITY` | `0.6` | ElevenLabs voice stability |
| `ELEVEN_SIMILARITY` | `0.75` | ElevenLabs similarity boost |
| `DEEPGRAM_API_KEY` | — | Uses Deepgram directly for STT when set |
| `CARTESIA_API_KEY` | — | Uses Cartesia directly for TTS when set |
| `OPENAI_API_KEY` | — | Used for LLM, analysis and judges when Groq is absent |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Conversation model |
| `GROQ_REASONING_EFFORT` | `low` | Keeps a reasoning model responsive on a call |
| `ANALYSIS_MODEL` | `openai/gpt-oss-120b` on Groq | Post-call analysis model |
| `OPIK_JUDGE_MODEL` | `groq/openai/gpt-oss-120b` | LLM-as-judge model |
| `OPIK_JUDGE_BACKEND` | `portable` | `portable` or `geval`, see above |
| `LIVEKIT_STT_MODEL` | `deepgram/nova-3-medical` | LiveKit Inference speech-to-text |
| `LIVEKIT_TTS_MODEL` | `cartesia/sonic-2` | LiveKit Inference text-to-speech |
| `STT_PROVIDER`, `LLM_PROVIDER`, `TTS_PROVIDER` | auto | Force a provider per stage |
| `AGENT_NAME` | `healthcare-outbound-agent` | Must match on worker and dispatcher |

### How a provider is chosen

Each pipeline stage resolves independently, in [agent.py](agent.py) `_resolve_provider`.
An explicit `STT_PROVIDER` / `LLM_PROVIDER` / `TTS_PROVIDER` always wins. Otherwise:

| Stage | Order |
|---|---|
| Speech to text | ElevenLabs key → Deepgram key → LiveKit Inference (`deepgram/nova-3-medical`) |
| Language model | Groq key → OpenAI key → LiveKit Inference |
| Text to speech | ElevenLabs key → Cartesia key → LiveKit Inference (`cartesia/sonic-2`) |

LiveKit Inference is speech served by LiveKit Cloud using the credentials the worker already
holds, so the agent runs with no model-provider key at all. The chosen providers are logged
at startup and recorded on the Opik trace, so a bad call can be tied back to a model.

Transcription accuracy is treated as a correctness problem, not a quality preference. A
mistranscribed biomarker name is the one error this agent cannot absorb. So the LiveKit
fallback uses `nova-3-medical`, and when ElevenLabs is configured its realtime transcriber
is seeded with keyterms built from the patient's own record: every biomarker label, the
recommended specialty, the care programme and their first name, plus a short clinical list.
See `_speech_keyterms` in [agent.py](agent.py).

---

## Tests

```bash
python -m pytest tests -q
```

40 tests, no network and no API keys. They cover patient lookup, metadata round-tripping,
the booking service including the rejected-slot path, every branch of the deterministic
analysis, the deterministic evaluators, provider resolution across Groq, OpenAI and LiveKit Inference,
the judge rate-limit retry and sequencing, .env placeholder handling,
and the exact Opik payload shape — the Opik client
is replaced with a recording double, so the tests assert which spans are created, that the
audio really becomes an `Attachment` with the right content type, that feedback scores land
on the trace, that `finalize()` is idempotent, and that a failing judge does not take down
the rest.

---

## Design decisions

**Booking truth comes from the tool record, not the LLM.** The single most important fact
about a healthcare call is whether an appointment exists. Asking a model to infer that from
a transcript introduces a failure mode with no upside, so `book_appointment` returning
`status: confirmed` is the only thing that counts. The LLM's opinion is kept and flagged
when it disagrees, which turns a silent error into a measurable one.

**The patient record is passed to the judge as ground truth.** `biomarker_fidelity` is a
grounded check, not a vibe check. The judge sees the real numbers and the agent's actual
words, so it can catch a misread value rather than just rating fluency.

**Observability can never break the call.** Every public entry point in
`opik_integration.py` is wrapped. Missing credentials produce a disabled no-op observer, a
crashing evaluator is dropped and the others still score, and a failed trace write is
logged rather than raised. A monitoring system that can hang up on a patient is worse than
no monitoring.

**The judges run sequentially, not in parallel.** Each one sends the transcript plus the
patient record, and firing them together exceeds the tokens-per-minute cap on a free
provider tier. They run after the call has ended, so a few extra seconds cost nothing, and
a rate limit is retried using the delay the provider states rather than a guess.

**Opik's SDK is synchronous, the audio path is not.** All blocking Opik work runs through
`asyncio.to_thread` so it cannot stall the realtime event loop.

**Traces are closed by re-sending them whole, not by `trace.end()`.** The Opik SDK batches
outbound messages, and a partial update sent close to the create can be dropped. That is not
theoretical: the first working version of this integration wrote traces that arrived with no
`end_time`, no output and none of the outcome tags, because the `trace.end()` update was
swallowed. Spans now carry their `end_time` at creation and are never updated, and the trace
is re-sent under the same id as one complete payload, which the backend applies wholesale.
The call recording rides on that final payload, so the audio is attached to the call itself
rather than buried in a span.

**Perceived latency on a phone call was almost entirely transcription.** Measuring a real
call rather than guessing gave transcription 3.9s per turn against 0.28s for speech
synthesis. ElevenLabs `scribe_v1` does not stream: it waits for you to stop talking, then
transcribes the whole utterance. Switching to `scribe_v2_realtime`, trimming the endpointing
delay and enabling preemptive generation attacks the part that actually cost the time. The
per-turn numbers ride on every Opik trace, so this is measurable rather than a matter of
opinion.

**The agent greets the phone only after someone answers.** The session starts before dialling
so no audio is lost on pickup, which means `on_enter` fires while the line is still ringing.
Greeting there played the opening into a ringing phone: a real call showed 10.9s of playback
latency and the patient answered midway through the first sentence. The greeting now fires
after `wait_for_participant`, on both the phone and browser paths.

**The opening line is spoken verbatim, not generated.** `on_enter` uses `session.say` rather
than `generate_reply`, so every call starts identically with the clinic name and an identity
check. Left to generate its own opening, the model would sometimes run straight into the lab
results before the person had confirmed who they were, which is the exact failure
`pii_disclosure_control` exists to catch.

**The prompt carries a worked example.** Stage instructions alone were not enough: the model
followed the structure but stayed clinical, opening turns by reciting numbers and asking
"does that make sense". A short annotated example dialogue in the prompt fixed the register in
a way that more rules did not. Certain phrases are banned outright, including any mention of
diabetes, because labelling a condition is a clinician's job and the agent kept drifting into it.

**Warmth lives in the data as well as the prompt.** The first version of the patient record
described HbA1c as "in the diabetes range", and the agent dutifully read that out loud. The
`plain_language` field on every biomarker is now written the way a care coordinator would
actually say it.

**The prompt is staged, and the turn limit is absolute.** Early versions told the agent to
"keep turns short" and it monologued anyway, delivering every biomarker in one breath. The
prompt now walks through six numbered stages with an explicit STOP after each, and a hard
two-sentence ceiling. Delivery is also slowed to 0.85 speed: default text-to-speech pace is
noticeably too fast for someone absorbing a number like "seven point eight percent".

**Identity before disclosure.** The prompt forbids stating any health value before the
patient confirms who they are, and `pii_disclosure_control` measures whether that actually
happened. Prompt rules are hopes; evaluators are evidence.

**The booking backend can say no.** A simulated service that always succeeds tests nothing.
Slot `...-1200` is always taken, so the agent has to recover and offer an alternative, and
that recovery shows up in the trace.

**`AgentServer`, not `WorkerOptions`.** `WorkerOptions(entrypoint_fnc=...)` still works in
1.8 but is no longer the documented API. `AgentServer` with `@server.rtc_session(agent_name=...)`
is current, and the `agent_name` is what makes dispatch explicit so the worker only takes
jobs the dispatcher sends it instead of auto-joining every room.

**Per-turn latency, not the deprecated metrics event.** `metrics_collected` is deprecated in
1.8 and warns at runtime. Usage comes from `session_usage_updated`, which the SDK already
aggregates per model, and latency comes off `ChatMessage.metrics` on each turn.

---

## Troubleshooting

**`Failed to dispatch the agent`** — the worker is not running, or `AGENT_NAME` differs
between the worker and the dispatcher. Start `python agent.py dev` first.

**The agent never speaks** — check the worker logs for a missing model API key. In
simulation mode make sure you joined the room the dispatcher printed.

**`error creating SIP participant`** — the logs include the SIP status code. `403` is
usually trunk credentials, `404` a bad number format (use E.164, `+15551234567`), and a
timeout usually means the trunk's termination URI is wrong.

**No trace in Opik** — confirm `OPIK_API_KEY` and `OPIK_WORKSPACE`, and check that
`OPIK_PROJECT_NAME` matches the project you are looking at. The worker logs
`Opik trace opened for call ...` on success and warns loudly when Opik is unconfigured.

**Evaluation scores missing** — the LLM judges need `GROQ_API_KEY` or `OPENAI_API_KEY`.
Without one, the two deterministic evaluators still run and the three LLM judges are skipped
rather than failing.

**`rate_limit_exceeded` from Groq** — the free tier allows 8000 tokens per minute. The
judges run sequentially and retry with the wait the provider asks for, so this self-resolves
and you will just see `judge hit a provider rate limit, retrying in Ns` in the log. Running
all three demo scenarios back to back will trigger it a few times.

**`402 paid_plan_required` from ElevenLabs** — a free account cannot use shared library or
"professional" voices through the API, only "premade" ones. List what yours can use with
`curl -H "xi-api-key: $ELEVEN_API_KEY" https://api.elevenlabs.io/v2/voices` and pick one whose
category is `premade`.

**`model_not_found` from Groq** — the available models change. List yours with
`curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models` and set
`LLM_MODEL` to one of them.

**`WinError 32` during pip install** — OneDrive is locking the virtualenv. Create it
outside any synced folder.
