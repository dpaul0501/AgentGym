"""Light smoke tests for the Travel Sage example agent (per the user's choice: real smoke
coverage, not full red-green TDD per class, given presentation-deadline time pressure). The
offline tests need no network; the two integration tests make real Bedrock/local-Ollama calls."""

import pytest

from agentgym.core.protocols import Task
from agentgym.core.scope import GUARDRAILS, MEMORY
from examples.travel_sage_agent.agent import TravelSageAgent
from examples.travel_sage_agent.benchmark import TravelBenchmark
from examples.travel_sage_agent.tools import INJECTION_PAYLOAD, get_weather, search_flights, search_hotels


def test_search_flights_returns_verbose_mock_listings():
    flights = search_flights.func("SFO", "JFK", "2026-09-01")
    assert len(flights) == 30
    assert all("baggage_policy" in f for f in flights)


def test_search_hotels_includes_the_injection_payload():
    hotels = search_hotels.func("Paris", "2026-09-01", "2026-09-04")
    reviews = [h["review"] for h in hotels]
    assert INJECTION_PAYLOAD in reviews


def test_get_weather_returns_well_formed_mock_data():
    weather = get_weather.func("Paris")
    assert weather["city"] == "Paris"
    assert "forecast" in weather


def test_travel_benchmark_has_real_task_prompts():
    cases = TravelBenchmark().cases()
    assert len(cases) >= 2
    assert all(isinstance(c, Task) for c in cases)


def test_agent_consumes_declares_memory_and_guardrails_with_no_default():
    agent = TravelSageAgent(llm=object())  # consumes() doesn't touch the LLM, so a stub is fine
    consumed = agent.consumes()
    assert consumed == {MEMORY: None, GUARDRAILS: None}


@pytest.mark.integration
def test_real_bedrock_run_returns_a_valid_trace_with_an_llm_span():
    agent = TravelSageAgent()
    task = Task(task_id="weather_check", instruction="What is the weather in Paris?", metadata={})

    trace = agent.run(task, {})

    assert trace.task_id == "weather_check"
    assert any(span.kind == "llm" for span in trace.spans)
    assert trace.metadata["final_response"]


@pytest.mark.integration
def test_real_ollama_run_returns_a_valid_trace():
    from examples.travel_sage_agent.llm import build_ollama_llm

    agent = TravelSageAgent(llm=build_ollama_llm("qwen2.5:7b"))
    task = Task(task_id="weather_check", instruction="What is the weather in Paris?", metadata={})

    trace = agent.run(task, {})

    assert trace.task_id == "weather_check"
    assert any(span.kind == "llm" for span in trace.spans)
