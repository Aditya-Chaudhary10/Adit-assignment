# Demo workflow

What to show, in what order, and what to say. Roughly twelve minutes.

Before you start:

```powershell
& "$env:USERPROFILE\.venvs\adit-voice-agent\Scripts\Activate.ps1"
cd "$env:USERPROFILE\OneDrive\Desktop\Adit-assingment"
```

Have three things open: two terminals, a browser on your Opik project, and this repo
in an editor.

---

## Step 0 — Set the frame (30 seconds)

"An outbound voice agent calls a patient, explains lab results they haven't seen,
and books a doctor. Everything about the call goes to Opik, and seven evaluations
score it automatically."

Show the architecture diagram at the top of `README.md`. Point at the three lines
that wire in Opik. Say: "nothing else in the codebase imports Opik."

---

## Step 1 — A live call, end to end (5 minutes) ← THE CENTREPIECE

Terminal 1:

```powershell
python agent.py dev
```

Terminal 2:

```powershell
python dispatch_call.py --patient PT-10432 --simulate
python join_link.py
```

Open the printed link, allow the microphone. **Then actually book the appointment.**

Follow this path so every code path fires:

| They say | You say |
|---|---|
| "Am I speaking with Aditya?" | "Yes, speaking." |
| "...is now an okay time?" | "Yeah, go ahead." |
| "...one thing the doctor would like to look at" | "Okay, what is it?" |
| "Your HbA1c came back at seven point eight..." | "Is that bad?" |
| "...get you in with an endocrinologist" | "Yeah, alright." |
| offers two times | pick one |
| reads the confirmation number | "No, that's all. Thanks." |

Then say goodbye so the agent hangs up. **Do not just close the tab.** Hanging up
properly is what triggers the shutdown pipeline.

While it runs, point at the worker log:

```
pipeline: stt=elevenlabs | llm=groq (openai/gpt-oss-120b) | tts=elevenlabs
```

"Providers resolve per stage from whatever keys are present, and the choice is
recorded on the trace, so a bad call can be traced back to a model."

After hangup, the log shows the whole post-call pipeline:

```
call recording saved to recordings/<call-id>.ogg
post-call analysis: outcome=appointment_booked booked=True
Opik trace finalized (scores={...})
```

---

## Step 2 — The trace in Opik (3 minutes)

Open the trace you just made. Walk it top down.

- **Input** — the patient record and call variables the agent was given.
- **Output** — outcome, `appointment_booked: true`, the confirmation id, the summary.
- **Attachment** — play the call recording. It is the real audio.
- **Spans** — `conversation` with one child per turn, `tool::check_availability`,
  `tool::book_appointment` with its arguments and the confirmation it returned,
  `post_call_analysis`, `online_evaluation`.
- **Feedback scores** — five, each with the judge's written reason.
- **Tags** — `outcome:appointment_booked`, `appointment_booked`.

Say: "whether an appointment was booked is read from the tool record, not inferred
by the model. The LLM's opinion is kept and flagged when it disagrees."

---

## Step 3 — Prove the evaluation actually catches failures (2 minutes)

This is the part that separates a demo from a claim.

```powershell
python demo_pipeline.py --scenario hallucination
```

A seeded bad call where the agent invents a normal HbA1c and suggests stopping a
medication. Show the scores:

```
biomarker_fidelity    0.05
medical_safety        0.00
conversation_quality  0.85
```

Then compare against the good call: both 1.00.

Say: "the judges discriminate. Conversation quality stays high on the bad call
because that judge scores mechanics only. Accuracy is `biomarker_fidelity`'s job
and clinical overreach is `medical_safety`'s. Folding accuracy into every judge
would correlate the scores and lose the signal about *which* thing broke."

Optionally run `--all` to show all three scenarios side by side.

---

## Step 4 — The Opik module is genuinely standalone (1 minute)

Open `opik_integration.py`. Scroll to the imports: stdlib and `opik`, nothing else.

Show the three lines in `agent.py`. Show the server-side rules:

```powershell
python opik_integration.py
```

"Two LLM-as-judge rules created in Opik itself, so evaluation also runs server
side on any trace that arrives, including from other services."

---

## Step 5 — Engineering depth (2 minutes)

Have the Design decisions section of `README.md` open. Lead with these three,
because they are real bugs found by verification rather than design talk.

1. **Traces were silently losing data.** Closing a trace with `trace.end()` sends a
   partial update, and Opik's batching was dropping it. Traces arrived with no
   end time and no outcome. Now the trace is re-sent whole under the same id.
   Found by reading traces back through the API, not by trusting the write.

2. **Bookings were invisible.** LiveKit stringifies tool returns with `str()`, so a
   dict arrives as a Python repr with single quotes. Parsing only as JSON meant
   every successful booking looked like an opaque string and `appointment_booked`
   was permanently False. A real call that booked twice reported `False`. Fixed by
   falling back to `ast.literal_eval`, with regression tests.

3. **Observability was stalling the call.** Opening the Opik trace is a synchronous
   HTTP round trip and it blocked the audio loop for a second. There is now an
   async `astart`, and a test that fails if it ever runs on the event loop thread.

Then: `python -m pytest tests -q` → 46 passing, no network, no API keys.

---

## Step 6 — Be honest about the gap (30 seconds)

"The SIP path is written and follows LiveKit's documented outbound pattern,
including SIP status code handling, but I have not run it against a live trunk.
Everything you have seen is browser-simulated. I would not claim the telephony
works until I have heard a phone ring."

Reviewers respect this far more than discovering it themselves.

---

## If something goes wrong live

| Symptom | Fix |
|---|---|
| Agent not in the room | Room expired. Dispatch again, join within 30 minutes. |
| Call never connects | Stale workers stealing dispatches. Kill all, start one. |
| `rate_limit` in the log | Groq free tier. It retries itself, just narrate it. |
| Judges score nothing | `GROQ_API_KEY` missing. Deterministic two still run. |

Have `transcripts/` and `recordings/` open as a fallback. If the live call fails,
show a saved artifact instead and keep moving.
