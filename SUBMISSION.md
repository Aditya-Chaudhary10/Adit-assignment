# Submission checklist

Every line in the assignment brief, mapped to where it is satisfied and how it was
verified. The one genuine gap is called out at the bottom rather than buried.

Status key: **DONE** verified working  ·  **PARTIAL** works but with a caveat  ·  **GAP** not done

---

## 1. Outbound AI voice agent using LiveKit, healthcare use case

| Requirement | Status | Where | Evidence |
|---|---|---|---|
| Built on LiveKit | DONE | `agent.py` | `livekit-agents` 1.8.3, `AgentServer` + `@server.rtc_session` |
| Outbound, not inbound | PARTIAL | `agent.py`, `dispatch_call.py` | `CreateSIPParticipantRequest` with `wait_until_answered`. Code path written, never fired at a real phone. See the gap. |
| Agent is given a Name | DONE | `data/patients.json` | Travels as dispatch metadata |
| Agent is given a Phone number | DONE | `data/patients.json` | `--phone` overrides it per call |
| Agent is given health biomarkers | DONE | `data/patients.json` | HbA1c, fasting glucose, LDL, blood pressure, vitamin D, each with value, unit, reference range, status |
| Informs the person about their metrics | DONE | `build_instructions` in `agent.py` | Live transcripts in `transcripts/` |
| Attempts to schedule a doctor consultation | DONE | Stage 5 of the prompt | Agent drives to booking rather than waiting to be asked |
| Tool/function call simulates booking | DONE | `booking_service.py` | `check_availability` and `book_appointment`; one slot is always taken so the agent must recover |

Seven tools total: `check_availability`, `book_appointment`, `record_patient_decision`,
`flag_for_urgent_followup`, `transfer_to_human`, `detected_answering_machine`, `end_call`.

## 2. Post-call analysis

| Requirement | Status | Where |
|---|---|---|
| Runs after the call ends | DONE | `_finalize_call` in a LiveKit shutdown callback |
| Analyses the conversation | DONE | `post_call_analysis.py`, strict JSON schema output |
| Determines the outcome | DONE | Ten outcomes, from `appointment_booked` to `wrong_number` |
| Determines whether an appointment was booked | DONE | Read from the tool record, not inferred by an LLM, so it cannot hallucinate |

Also returns sentiment, objections, biomarkers actually spoken, agent-side quality issues,
escalation flag and a summary. Degrades to a deterministic answer if no LLM key is present.

## 3. Opik integration

| Requirement | Status | Evidence in a live trace |
|---|---|---|
| Call metadata and variables | DONE | Trace input: patient id, name, phone, clinic, specialty, room, job id, resolved providers and models |
| Conversation / transcript | DONE | `conversation` span plus one `turn_NN_<role>` child per utterance |
| Call recording or audio reference | DONE | Real `.ogg` attached to the trace. Largest so far 834 KB of genuine call audio |
| Tool calls / results | DONE | One `tool::<name>` span each, arguments and results |
| Post-call analysis | DONE | `post_call_analysis` span carrying the full structured object |
| At least one online evaluation | DONE | Five in-process evaluators, plus two server-side rules in Opik |
| Standalone, single modular file | DONE | `opik_integration.py` imports nothing from this project |
| Plugs in with minimal changes | DONE | Three lines in `agent.py`: `astart`, `attach`, `finalize` |

### The online evaluations

Five run in-process on every completed call and post feedback scores to the trace:

| Evaluator | Method |
|---|---|
| `appointment_conversion` | Deterministic, from the tool record |
| `pii_disclosure_control` | Deterministic, checks identity was confirmed before any value was spoken |
| `biomarker_fidelity` | LLM judge grounded on the patient record, catches hallucinated lab values |
| `medical_safety` | LLM judge, catches diagnosis, prescribing, dosage advice |
| `conversation_quality` | LLM judge, patient-experience view |

Two more run server-side inside Opik, created by `python opik_integration.py`:
`voice-agent-medical-safety` and `voice-agent-outcome-consistency`, both enabled at
sampling 1.0. Verified present via the Opik API.

The scores discriminate rather than rubber-stamp. On the seeded bad call the agent invents a
normal HbA1c and suggests stopping a medication; `biomarker_fidelity` scores 0.05 and
`medical_safety` 0.00, while clean calls score 1.00 on both.

## 4. Deliverables

| Deliverable | Status | Where |
|---|---|---|
| Working implementation | DONE | This repository |
| README with setup and usage | DONE | `README.md`, plus `RUN.md` for the short version |
| Demonstration of the complete flow | DONE | `demo_pipeline.py --all`, and live call artifacts in `transcripts/` and `recordings/` |
| Opik trace / evaluation for the call | DONE | 12 traces in project `healthcare-voice-agent` |
| Opik integration standalone and modular | DONE | `opik_integration.py` |

## 5. Able to explain the code and decisions

`README.md` carries a Design decisions section covering the choices a reviewer is most
likely to probe: why booking truth comes from the tool record, why the patient record is
handed to the judge as ground truth, why traces are closed by re-sending them whole rather
than with `trace.end()`, why the judges run sequentially, and why the prompt carries a worked
example. 41 offline tests, no network or API keys required.

---

## The one real gap

**No call has been placed to an actual telephone.** Every call so far has been simulated,
with a browser standing in for the patient. The SIP code path is written and matches
LiveKit's documented outbound pattern, including SIP status code handling on failure, but it
has never executed against a live trunk, so I cannot claim it works.

To close it:

1. Buy a voice-capable number from Twilio, Telnyx or Plivo.
2. Create an Elastic SIP Trunk and note the termination URI and SIP credentials.
3. Register it with LiveKit: `lk sip outbound create outbound-trunk.json`
4. Put the returned id in `.env` as `SIP_OUTBOUND_TRUNK_ID`.
5. `python dispatch_call.py --patient PT-10432 --phone +91XXXXXXXXXX`

Full walkthrough in the README under "Placing a real phone call". Roughly fifteen minutes and
a couple of dollars. Until then, describe the telephony layer as implemented but untested.

## Smaller caveats worth knowing before a review

- **No live call has completed a booking end to end.** The booking tool is exercised in the
  scripted demo and was invoked in one live call, but no browser call has run all the way to a
  confirmed appointment. Worth doing one clean run before submitting.
- **Opik's G-Eval is not the judge in use.** It requires `logprobs`, which Groq's models do
  not support, so the LLM judges use a portable strict-JSON judge instead. `OPIK_JUDGE_BACKEND=geval`
  switches to Opik's own metric on a provider that supports it.
- **Groq's free tier rate-limits the judges.** They run sequentially and retry on the delay
  the provider states. Expect retry lines in the log during a full demo run.
