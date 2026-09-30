"""Offline tests for the post-call pipeline and the Opik integration.

Run with:  python -m pytest tests -q

The Opik client is replaced with a recording double, so these tests verify the
exact shape of what would be sent to Opik without needing an API key or network.
"""

from __future__ import annotations

import asyncio
import os

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import booking_service  # noqa: E402
import opik_integration  # noqa: E402
from call_data import CallContext, find_patient  # noqa: E402
from post_call_analysis import analyze_call  # noqa: E402


# --------------------------------------------------------------------------
# Opik test double
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_booking_store(tmp_path, monkeypatch):
    """Keep test bookings out of data/bookings.json."""
    monkeypatch.setattr(booking_service, "BOOKINGS_FILE", str(tmp_path / "bookings.json"))


class FakeSpan:
    def __init__(self, recorder: list, **kwargs: Any) -> None:
        self._recorder = recorder
        self.kwargs = kwargs
        recorder.append(("span", kwargs))

    def span(self, **kwargs: Any) -> "FakeSpan":
        return FakeSpan(self._recorder, **kwargs)

    def end(self, **kwargs: Any) -> None:
        self._recorder.append(("span_end", kwargs))

    def log_feedback_score(self, **kwargs: Any) -> None:
        self._recorder.append(("span_score", kwargs))


class FakeTrace:
    _next_id = 0

    def __init__(self, recorder: list, **kwargs: Any) -> None:
        self._recorder = recorder
        FakeTrace._next_id += 1
        self.id = f"fake-trace-{FakeTrace._next_id}"
        recorder.append(("trace", kwargs))

    def span(self, **kwargs: Any) -> FakeSpan:
        return FakeSpan(self._recorder, **kwargs)

    def update(self, **kwargs: Any) -> None:
        self._recorder.append(("trace_update", kwargs))

    def end(self, **kwargs: Any) -> None:
        self._recorder.append(("trace_end", kwargs))

    def log_feedback_score(self, **kwargs: Any) -> None:
        self._recorder.append(("feedback_score", kwargs))


class FakeClient:
    def __init__(self, recorder: list) -> None:
        self._recorder = recorder
        self.flushed = False

    def flush(self) -> None:
        self.flushed = True
        self._recorder.append(("flush", {}))

    def trace(self, **kwargs):
        self._recorder.append(("trace_upsert", kwargs))
        return FakeTrace(self._recorder, **kwargs)


def build_observer(recorder: list, **overrides: Any) -> opik_integration.OpikCallObserver:
    patient = find_patient("PT-10432")
    return opik_integration.OpikCallObserver(
        client=FakeClient(recorder),
        trace=FakeTrace(recorder, name="test"),
        call_id=overrides.get("call_id", "test-call-1"),
        variables=overrides.get("variables", {"patient_id": patient.patient_id}),
        grounding=overrides.get("grounding", patient.to_dict()),
        evaluators=overrides.get("evaluators", [opik_integration.eval_appointment_conversion,
                                                opik_integration.eval_pii_disclosure_control]),
    )


TRANSCRIPT_BOOKED = [
    {"role": "assistant", "content": "Hi, this is Riley from the clinic. Am I speaking with Aditya?"},
    {"role": "user", "content": "Yes, speaking."},
    {"role": "assistant", "content": "Your HbA1c is 7.8 percent, above the 4.0 - 5.6 range."},
    {"role": "user", "content": "Okay, book me in."},
]

TOOLS_BOOKED = [
    {
        "name": "book_appointment",
        "call_id": "c1",
        "arguments": {"date": "2026-10-06", "time_of_day": "10:00 AM"},
        "output": {"status": "confirmed", "confirmation_id": "APT-441209"},
    }
]


# --------------------------------------------------------------------------
# Domain model
# --------------------------------------------------------------------------


def test_patient_lookup_by_id_name_and_phone():
    assert find_patient("PT-10432").name == "Aditya Chaudhary"
    assert find_patient("meera").patient_id == "PT-20871"
    assert find_patient("+919999999999").patient_id == "PT-10432"


def test_unknown_patient_raises():
    with pytest.raises(KeyError):
        find_patient("nobody-here")


def test_call_context_metadata_roundtrip():
    patient = find_patient("PT-10432")
    ctx = CallContext(patient=patient, call_id="abc", transfer_to="+15550001111", simulate=True)
    restored = CallContext.from_metadata(ctx.to_metadata())
    assert restored.call_id == "abc"
    assert restored.simulate is True
    assert restored.transfer_to == "+15550001111"
    assert restored.patient.biomarkers[0].label == patient.biomarkers[0].label


def test_abnormal_biomarkers_detected():
    patient = find_patient("PT-10432")
    keys = {b.key for b in patient.abnormal_biomarkers}
    assert "hba1c" in keys
    assert len(keys) == 4


# --------------------------------------------------------------------------
# Booking service
# --------------------------------------------------------------------------


def test_booking_confirms_and_returns_confirmation_id():
    result = booking_service.book(
        patient_id="PT-TEST", patient_name="Test", date="2026-10-06",
        time_of_day="10:00 AM", slot_id="20261006-1000", specialty="Endocrinology",
    )
    assert result["status"] == "confirmed"
    assert result["confirmation_id"].startswith("APT-")


def test_booking_rejects_taken_slot():
    result = booking_service.book(
        patient_id="PT-TEST", patient_name="Test", date="2026-10-06",
        time_of_day="12:00 PM", slot_id="20261006-1200",
    )
    assert result["status"] == "rejected"
    assert result["reason"] == "slot_taken"


def test_booking_requires_a_slot():
    result = booking_service.book(patient_id="PT-TEST", patient_name="Test")
    assert result["status"] == "rejected"
    assert result["reason"] == "missing_slot"


def test_availability_never_offers_blocked_hours():
    slots = booking_service.get_availability("Endocrinology")["slots"]
    assert slots
    for s in slots:
        assert not s["slot_id"].endswith(("-0800", "-1300", "-1800"))


# --------------------------------------------------------------------------
# Post-call analysis
# --------------------------------------------------------------------------


def test_analysis_reads_booking_from_tool_record(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = asyncio.run(
        analyze_call(patient={}, transcript=TRANSCRIPT_BOOKED, tool_calls=TOOLS_BOOKED)
    )
    assert result["appointment_booked"] is True
    assert result["outcome"] == "appointment_booked"
    assert result["appointment_details"]["confirmation_id"] == "APT-441209"
    assert result["analysis_mode"] == "deterministic_fallback"


def test_analysis_marks_no_answer_when_patient_never_spoke(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = asyncio.run(
        analyze_call(
            patient={}, transcript=[{"role": "assistant", "content": "Hello?"}], tool_calls=[]
        )
    )
    assert result["outcome"] == "no_answer"
    assert result["appointment_booked"] is False


def test_analysis_ignores_a_rejected_booking(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tools = [{"name": "book_appointment", "arguments": {},
              "output": {"status": "rejected", "reason": "slot_taken"}}]
    result = asyncio.run(analyze_call(patient={}, transcript=TRANSCRIPT_BOOKED, tool_calls=tools))
    assert result["appointment_booked"] is False
    assert result["outcome"] == "needs_human_followup"


def test_analysis_parses_stringified_tool_output(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tools = [{"name": "book_appointment", "arguments": {},
              "output": '{"status": "confirmed", "confirmation_id": "APT-1"}'}]
    result = asyncio.run(analyze_call(patient={}, transcript=TRANSCRIPT_BOOKED, tool_calls=tools))
    assert result["appointment_booked"] is True


# --------------------------------------------------------------------------
# Online evaluators
# --------------------------------------------------------------------------


def _eval_input(**overrides: Any) -> opik_integration.EvaluationInput:
    patient = find_patient("PT-10432")
    base = {
        "transcript": TRANSCRIPT_BOOKED,
        "tool_calls": TOOLS_BOOKED,
        "analysis": {"appointment_booked": True},
        "variables": {},
        "grounding": patient.to_dict(),
        "telemetry": {},
    }
    base.update(overrides)
    return opik_integration.EvaluationInput(**base)


def test_conversion_scores_one_when_booked():
    r = asyncio.run(opik_integration.eval_appointment_conversion(_eval_input()))
    assert r.value == 1.0
    assert r.category == "booked"


def test_conversion_scores_half_when_only_attempted():
    inp = _eval_input(
        analysis={"appointment_booked": False},
        tool_calls=[{"name": "book_appointment", "arguments": {},
                     "output": {"status": "rejected"}}],
    )
    r = asyncio.run(opik_integration.eval_appointment_conversion(inp))
    assert r.value == 0.5
    assert r.category == "attempted"


def test_conversion_scores_zero_when_never_attempted():
    r = asyncio.run(opik_integration.eval_appointment_conversion(
        _eval_input(analysis={"appointment_booked": False}, tool_calls=[])
    ))
    assert r.value == 0.0


def test_pii_control_passes_when_identity_confirmed_first():
    r = asyncio.run(opik_integration.eval_pii_disclosure_control(_eval_input()))
    assert r.value == 1.0
    assert r.category == "verified_first"


def test_pii_control_fails_when_values_leak_before_verification():
    leaky = [
        {"role": "assistant", "content": "Hi, your HbA1c is 7.8 percent and it's high."},
        {"role": "user", "content": "Who is this?"},
    ]
    r = asyncio.run(opik_integration.eval_pii_disclosure_control(_eval_input(transcript=leaky)))
    assert r.value == 0.0
    assert r.category == "disclosed_without_verification"


# --------------------------------------------------------------------------
# Opik observer wiring
# --------------------------------------------------------------------------


def test_observer_is_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    monkeypatch.delenv("OPIK_URL_OVERRIDE", raising=False)
    obs = opik_integration.OpikCallObserver.start(call_id="x")
    assert obs.enabled is False
    # finalize must be a safe no-op rather than an exception
    result = asyncio.run(obs.finalize(analysis={"outcome": "no_answer"}))
    assert result["enabled"] is False


def test_finalize_writes_expected_spans_and_scores(tmp_path):
    recorder: list = []
    obs = build_observer(recorder)
    audio = tmp_path / "call.ogg"
    audio.write_bytes(b"not really ogg, but a real file")

    result = asyncio.run(
        obs.finalize(
            analysis={
                "outcome": "appointment_booked", "appointment_booked": True,
                "summary": "Booked.", "analysis_mode": "llm", "analysis_model": "gpt-4o-mini",
                "appointment_details": {"confirmation_id": "APT-441209"},
            },
            audio_path=str(audio),
            telemetry={"duration_s": 100},
            transcript=TRANSCRIPT_BOOKED,
            tool_calls=TOOLS_BOOKED,
        )
    )
    assert result["ok"] is True

    span_names = [k.get("name") for kind, k in recorder if kind == "span"]
    assert "conversation" in span_names
    assert "tool::book_appointment" in span_names
    assert "call_recording" in span_names
    assert "post_call_analysis" in span_names
    assert "online_evaluation" in span_names
    assert sum(1 for n in span_names if n and n.startswith("turn_")) == len(TRANSCRIPT_BOOKED)

    # The audio file became a real Opik attachment on the trace itself.
    upsert = next(k for kind, k in recorder if kind == "trace_upsert")
    assert upsert["attachments"], "expected the recording to be attached to the trace"
    assert upsert["attachments"][0].file_name.endswith(".ogg")
    assert upsert["attachments"][0].content_type == "audio/ogg"
    rec_span = next(k for kind, k in recorder if kind == "span" and k.get("name") == "call_recording")
    assert rec_span["metadata"]["attached_to"] == "trace"

    # Feedback scores landed on the trace.
    scores = {k["name"]: k["value"] for kind, k in recorder if kind == "feedback_score"}
    assert scores["appointment_conversion"] == 1.0
    assert scores["pii_disclosure_control"] == 1.0

    # The trace is re-sent once as a complete payload, not partially updated.
    upserts = [k for kind, k in recorder if kind == "trace_upsert"]
    assert len(upserts) == 1
    up = upserts[0]
    assert up["id"] is not None
    assert up["end_time"] is not None
    assert up["input"] is not None, "the upsert must carry the original input"
    assert up["output"]["appointment_booked"] is True
    assert "outcome:appointment_booked" in up["tags"]
    assert "appointment_booked" in up["tags"]
    assert up["metadata"]["status"] == "completed"
    assert not any(kind == "trace_end" for kind, _ in recorder), (
        "trace.end() sends a partial update that batching can drop"
    )
    assert any(kind == "flush" for kind, _ in recorder)

    # Spans must be written complete in one message. Calling .end() on a span
    # shortly after creating it can be dropped by the SDK's batching.
    assert not any(kind == "span_end" for kind, _ in recorder), (
        "spans should carry end_time at creation, not be .end()-ed afterwards"
    )
    assert not any(kind == "trace_update" for kind, _ in recorder)
    for kind, kwargs in recorder:
        if kind == "span":
            assert kwargs.get("end_time") is not None, f"span {kwargs.get('name')} has no end_time"


def test_finalize_is_idempotent():
    recorder: list = []
    obs = build_observer(recorder)
    asyncio.run(obs.finalize(analysis={"outcome": "no_answer"}))
    before = len(recorder)
    asyncio.run(obs.finalize(analysis={"outcome": "no_answer"}))
    assert len(recorder) == before, "finalize must not write the trace twice"


def test_finalize_without_audio_records_a_note():
    recorder: list = []
    obs = build_observer(recorder)
    asyncio.run(obs.finalize(analysis={"outcome": "no_answer"}, transcript=[], tool_calls=[]))
    rec_span = next(k for kind, k in recorder if kind == "span" and k.get("name") == "call_recording")
    assert "note" in rec_span["metadata"]
    assert rec_span["metadata"]["attached_to"] == "nothing"
    upsert = next(k for kind, k in recorder if kind == "trace_upsert")
    assert upsert["attachments"] is None


def test_a_failing_evaluator_does_not_break_finalize():
    async def exploding(_inp):
        raise RuntimeError("judge is down")

    recorder: list = []
    obs = build_observer(recorder, evaluators=[exploding, opik_integration.eval_appointment_conversion])
    result = asyncio.run(
        obs.finalize(analysis={"appointment_booked": True}, transcript=TRANSCRIPT_BOOKED,
                     tool_calls=TOOLS_BOOKED)
    )
    assert result["ok"] is True
    assert "appointment_conversion" in result["scores"]


# --------------------------------------------------------------------------
# Live session event capture
# --------------------------------------------------------------------------


class _Item:
    def __init__(self, role: str, text: str, metrics: dict | None = None) -> None:
        self.role = role
        self.text_content = text
        self.interrupted = False
        self.metrics = metrics or {}


class _Ev:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def test_observer_captures_turns_and_latency():
    obs = build_observer([])
    obs._on_conversation_item(_Ev(item=_Item("assistant", "Hello", {"llm_node_ttft": 0.42}), created_at=1.0))
    obs._on_conversation_item(_Ev(item=_Item("user", "Hi there", {"end_of_turn_delay": 0.3}), created_at=2.0))
    obs._on_conversation_item(_Ev(item=_Item("assistant", "   ", {}), created_at=3.0))  # ignored

    assert len(obs.transcript) == 2
    assert obs.transcript[0]["content"] == "Hello"
    summary = obs.usage_summary()
    assert summary["llm_ttft_avg_s"] == 0.42
    assert summary["end_of_turn_delay_avg_s"] == 0.3
    assert summary["turns"] == 2


def test_observer_captures_tool_calls():
    obs = build_observer([])
    call = _Ev(name="book_appointment", call_id="c9", arguments='{"date": "2026-10-06"}')
    out = _Ev(call_id="c9", output='{"status": "confirmed"}', is_error=False)
    obs._on_tools_executed(_Ev(function_calls=[call], function_call_outputs=[out]))

    assert len(obs.tool_calls) == 1
    captured = obs.tool_calls[0]
    assert captured["name"] == "book_appointment"
    assert captured["arguments"] == {"date": "2026-10-06"}
    assert captured["output"] == {"status": "confirmed"}


def test_observer_captures_usage_and_errors():
    obs = build_observer([])
    usage = _Ev(model_usage=[_Ev(model_dump=lambda: {"type": "llm_usage", "model": "gpt-4o-mini",
                                                     "input_tokens": 100, "output_tokens": 0})])
    obs._on_usage_updated(_Ev(usage=usage))
    obs._on_error(_Ev(error="stt exploded", source=object()))

    summary = obs.usage_summary()
    assert summary["model_usage"][0]["model"] == "gpt-4o-mini"
    assert "output_tokens" not in summary["model_usage"][0]  # zeros are pruned
    assert summary["errors"] == 1


def test_observer_event_handlers_never_raise():
    obs = build_observer([])
    # Deliberately malformed payloads: capture must swallow them.
    obs._on_conversation_item(_Ev())
    obs._on_tools_executed(_Ev())
    obs._on_usage_updated(_Ev())
    obs._on_error(_Ev())
    obs._on_close(_Ev())
    assert obs.transcript == []


# --------------------------------------------------------------------------
# Provider resolution
# --------------------------------------------------------------------------


def _clear_provider_env(monkeypatch):
    for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY",
              "ELEVEN_API_KEY", "ELEVENLABS_API_KEY",
              "STT_PROVIDER", "LLM_PROVIDER", "TTS_PROVIDER", "OPIK_JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)


def test_llm_prefers_openai_then_groq_then_livekit(monkeypatch):
    """OpenAI first: Groq's gpt-oss models reason before answering, heard as dead air."""
    import agent

    _clear_provider_env(monkeypatch)
    assert agent._resolve_provider("llm") == "livekit"

    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    assert agent._resolve_provider("llm") == "groq"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert agent._resolve_provider("llm") == "openai"


def test_speech_falls_back_to_livekit_inference(monkeypatch):
    import agent

    _clear_provider_env(monkeypatch)
    assert agent._resolve_provider("stt") == "livekit"
    assert agent._resolve_provider("tts") == "livekit"

    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    monkeypatch.setenv("CARTESIA_API_KEY", "ct")
    assert agent._resolve_provider("stt") == "deepgram"
    assert agent._resolve_provider("tts") == "cartesia"


def test_explicit_provider_override_wins(monkeypatch):
    import agent

    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert agent._resolve_provider("llm") == "groq"


def test_judge_model_follows_the_available_key(monkeypatch):
    _clear_provider_env(monkeypatch)
    assert opik_integration.judges_available() is False

    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    assert opik_integration.judge_model() == "groq/openai/gpt-oss-120b"
    assert opik_integration.judges_available() is True

    monkeypatch.setenv("OPIK_JUDGE_MODEL", "groq/llama-3.3-70b-versatile")
    assert opik_integration.judge_model() == "groq/llama-3.3-70b-versatile"


def test_analysis_provider_prefers_groq_and_uses_a_structured_output_model(monkeypatch):
    from post_call_analysis import resolve_analysis_provider

    _clear_provider_env(monkeypatch)
    monkeypatch.delenv("ANALYSIS_MODEL", raising=False)
    assert resolve_analysis_provider() is None

    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    provider, key, base_url, model = resolve_analysis_provider()
    assert (provider, model) == ("openai", "gpt-4o-mini")

    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    provider, key, base_url, model = resolve_analysis_provider()
    assert provider == "groq"
    assert base_url == "https://api.groq.com/openai/v1"
    # Must be a Groq model that supports strict json_schema output.
    assert model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")


def test_llm_judges_skip_without_a_key(monkeypatch):
    _clear_provider_env(monkeypatch)
    patient = find_patient("PT-10432")
    inp = opik_integration.EvaluationInput(
        transcript=TRANSCRIPT_BOOKED, tool_calls=TOOLS_BOOKED, analysis={},
        variables={}, grounding=patient.to_dict(), telemetry={},
    )
    assert asyncio.run(opik_integration.eval_medical_safety(inp)) is None
    assert asyncio.run(opik_integration.eval_biomarker_fidelity(inp)) is None


# --------------------------------------------------------------------------
# Environment loading
# --------------------------------------------------------------------------


def test_template_placeholders_are_ignored(monkeypatch, tmp_path):
    from call_data import load_environment

    env = tmp_path / ".env"
    env.write_text(
        "GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx\n"
        "OPIK_API_KEY=your-opik-api-key\n"
        "LIVEKIT_API_KEY=APIrealkey123\n",
        encoding="utf-8",
    )
    for k in ("GROQ_API_KEY", "OPIK_API_KEY", "LIVEKIT_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    ignored = load_environment(str(env))
    assert set(ignored) == {"GROQ_API_KEY", "OPIK_API_KEY"}
    assert os.environ.get("GROQ_API_KEY") is None
    assert os.environ["LIVEKIT_API_KEY"] == "APIrealkey123"


def test_surrounding_whitespace_and_quotes_are_stripped(monkeypatch, tmp_path):
    from call_data import load_environment

    env = tmp_path / ".env"
    env.write_text('GROQ_API_KEY="gsk_realkeyvalue123 "\n', encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    load_environment(str(env))
    assert os.environ["GROQ_API_KEY"] == "gsk_realkeyvalue123"


# --------------------------------------------------------------------------
# Judge rate-limit handling
# --------------------------------------------------------------------------


class _FakeLiteLLM:
    """Fails with a provider rate-limit error N times, then succeeds."""

    def __init__(self, fail_times: int, message: str) -> None:
        self.fail_times = fail_times
        self.message = message
        self.calls = 0

    async def acompletion(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(self.message)
        return {"ok": True}


def test_judge_retries_rate_limits_using_the_stated_delay(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(opik_integration.asyncio, "sleep", fake_sleep)
    fake = _FakeLiteLLM(
        2, 'rate_limit_exceeded: Limit 8000. Please try again in 4.5s. code: rate_limit_exceeded'
    )
    result = asyncio.run(opik_integration._acompletion_with_backoff(fake, model="m", messages=[]))

    assert result == {"ok": True}
    assert fake.calls == 3
    # Honours the provider's stated wait rather than guessing.
    assert slept == [5.0, 5.0]


def test_judge_does_not_retry_other_errors(monkeypatch):
    async def fake_sleep(seconds):
        raise AssertionError("should not sleep for a non-rate-limit error")

    monkeypatch.setattr(opik_integration.asyncio, "sleep", fake_sleep)
    fake = _FakeLiteLLM(1, "invalid api key")
    with pytest.raises(RuntimeError, match="invalid api key"):
        asyncio.run(opik_integration._acompletion_with_backoff(fake, model="m", messages=[]))
    assert fake.calls == 1


def test_judge_gives_up_after_the_attempt_budget(monkeypatch):
    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(opik_integration.asyncio, "sleep", fake_sleep)
    fake = _FakeLiteLLM(99, "429 rate_limit_exceeded")
    with pytest.raises(RuntimeError):
        asyncio.run(
            opik_integration._acompletion_with_backoff(fake, attempts=3, model="m", messages=[])
        )
    assert fake.calls == 3


def test_evaluators_run_sequentially_not_concurrently():
    """Concurrent judges blow the tokens-per-minute cap on a free provider tier."""
    order: list[str] = []

    def make(name: str):
        async def _ev(_inp):
            order.append(f"start:{name}")
            await asyncio.sleep(0)
            order.append(f"end:{name}")
            return opik_integration.EvaluationResult(name=name, value=1.0)
        _ev.__name__ = f"eval_{name}"
        return _ev

    recorder: list = []
    obs = build_observer(recorder, evaluators=[make("a"), make("b")])
    asyncio.run(obs.finalize(analysis={}, transcript=[], tool_calls=[]))

    assert order == ["start:a", "end:a", "start:b", "end:b"]


def test_deepgram_transcribes_and_elevenlabs_speaks(monkeypatch):
    import agent

    _clear_provider_env(monkeypatch)
    monkeypatch.delenv("ELEVEN_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert agent._resolve_provider("stt") == "livekit"

    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    monkeypatch.setenv("CARTESIA_API_KEY", "ct")
    assert agent._resolve_provider("stt") == "deepgram"
    assert agent._resolve_provider("tts") == "cartesia"

    # ElevenLabs takes over speech synthesis, but Deepgram keeps transcription:
    # that stage is latency-critical and measured far faster on Deepgram.
    monkeypatch.setenv("ELEVEN_API_KEY", "el")
    assert agent._resolve_provider("stt") == "deepgram"
    assert agent._resolve_provider("tts") == "elevenlabs"
    # ElevenLabs does not serve the language model.
    assert agent._resolve_provider("llm") == "livekit"


def test_keyterms_come_from_the_patient_record():
    import agent

    terms = agent._speech_keyterms(find_patient("PT-10432"))
    assert "HbA1c" in terms
    assert "Endocrinology" in terms
    assert "Aditya" in terms
    assert all(t == t.strip() and t for t in terms)
    assert len(terms) == len(set(terms))


def test_astart_does_not_block_the_event_loop(monkeypatch):
    """The agent opens its trace from a realtime loop, so it must not block."""
    import threading

    calls = {}

    def slow_start(**kwargs):
        calls["thread"] = threading.current_thread().name
        return opik_integration.OpikCallObserver(
            client=None, trace=None, call_id=kwargs.get("call_id", "x"),
            variables={}, grounding={},
        )

    monkeypatch.setattr(opik_integration.OpikCallObserver, "start", staticmethod(slow_start))

    async def run():
        main = threading.current_thread().name
        obs = await opik_integration.OpikCallObserver.astart(call_id="abc")
        return main, obs

    main_thread, obs = asyncio.run(run())
    assert obs.call_id == "abc"
    assert calls["thread"] != main_thread, "start() must run off the event loop thread"


# --------------------------------------------------------------------------
# Tool output parsing
#
# Regression: LiveKit stringifies tool returns with str(), so a dict arrives as
# a Python repr with single quotes. Parsing only as JSON made every real booking
# invisible and appointment_booked was permanently False.
# --------------------------------------------------------------------------


PY_REPR_CONFIRMED = (
    "{'status': 'confirmed', 'confirmation_id': 'APT-233681', "
    "'patient_id': 'PT-10432', 'date': '2026-09-29', 'time': '11:00 AM'}"
)


def test_booking_detected_from_python_repr_output(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    tools = [{"name": "book_appointment", "arguments": {}, "output": PY_REPR_CONFIRMED}]
    result = asyncio.run(analyze_call(patient={}, transcript=TRANSCRIPT_BOOKED, tool_calls=tools))
    assert result["appointment_booked"] is True
    assert result["outcome"] == "appointment_booked"
    assert result["appointment_details"]["confirmation_id"] == "APT-233681"


def test_booking_detected_across_all_output_shapes(monkeypatch):
    from post_call_analysis import coerce_tool_output

    assert coerce_tool_output(PY_REPR_CONFIRMED)["status"] == "confirmed"
    assert coerce_tool_output('{"status": "confirmed"}')["status"] == "confirmed"
    assert coerce_tool_output({"status": "confirmed"})["status"] == "confirmed"
    # Junk must not raise, and must not be mistaken for a confirmation.
    assert coerce_tool_output("not a dict at all") == "not a dict at all"
    assert coerce_tool_output("{oh no") == {"raw": "{oh no"}


def test_a_later_rejection_does_not_undo_an_earlier_confirmation(monkeypatch):
    """A real call booked, hit a taken slot on a retry, then booked again."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    tools = [
        {"name": "book_appointment", "arguments": {},
         "output": "{'status': 'confirmed', 'confirmation_id': 'APT-1'}"},
        {"name": "book_appointment", "arguments": {},
         "output": "{'status': 'rejected', 'reason': 'slot_taken'}"},
        {"name": "book_appointment", "arguments": {},
         "output": "{'status': 'confirmed', 'confirmation_id': 'APT-2'}"},
    ]
    result = asyncio.run(analyze_call(patient={}, transcript=TRANSCRIPT_BOOKED, tool_calls=tools))
    assert result["appointment_booked"] is True
    assert result["appointment_details"]["confirmation_id"] == "APT-2"


def test_outcome_never_contradicts_the_booking_flag(monkeypatch):
    """A trace saying booked=False with outcome=appointment_booked looks like corruption."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    tools = [{"name": "book_appointment", "arguments": {},
              "output": "{'status': 'rejected', 'reason': 'slot_taken'}"}]
    result = asyncio.run(analyze_call(patient={}, transcript=TRANSCRIPT_BOOKED, tool_calls=tools))
    assert result["appointment_booked"] is False
    assert result["outcome"] != "appointment_booked"


def test_observer_parses_python_repr_tool_output():
    obs = build_observer([])
    call = _Ev(name="book_appointment", call_id="c1", arguments="{'date': '2026-09-29'}")
    out = _Ev(call_id="c1", output=PY_REPR_CONFIRMED, is_error=False)
    obs._on_tools_executed(_Ev(function_calls=[call], function_call_outputs=[out]))

    captured = obs.tool_calls[0]
    assert captured["arguments"] == {"date": "2026-09-29"}
    assert captured["output"]["confirmation_id"] == "APT-233681"


def test_keyterms_respect_the_provider_length_limit():
    """ElevenLabs realtime drops the whole STT websocket on an over-long keyterm.

    A live call died on "Fasting blood glucose", 21 characters, mid-conversation.
    """
    import agent

    terms = agent._speech_keyterms(find_patient("PT-10432"))
    assert terms, "expected some keyterms"
    over = [t for t in terms if len(t) > agent.MAX_KEYTERM_CHARS]
    assert not over, f"keyterms exceed the {agent.MAX_KEYTERM_CHARS} char limit: {over}"
    # The long label is dropped but its distinctive words survive.
    assert "Fasting blood glucose" not in terms
    assert "glucose" in terms
    assert "HbA1c" in terms
