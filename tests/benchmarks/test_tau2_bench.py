"""Tests against real tau2-bench task data (agentgym/benchmarks/data/tau2_mock_tasks.json,
sourced directly from sierra-research/tau2-bench) — not a synthetic fixture."""

from agentgym.benchmarks.tau2_bench import Tau2Benchmark


def test_loads_real_mock_domain_tasks():
    benchmark = Tau2Benchmark(domain="mock")
    cases = benchmark.cases()
    assert len(cases) == 10  # the real, current count in sierra-research/tau2-bench's mock domain


def test_real_task_has_expected_id_and_instruction():
    benchmark = Tau2Benchmark(domain="mock")
    cases = {t.task_id: t for t in benchmark.cases()}
    task = cases["create_task_1"]
    assert "Important Meeting" in task.instruction


def test_real_task_carries_real_evaluation_criteria():
    benchmark = Tau2Benchmark(domain="mock")
    cases = {t.task_id: t for t in benchmark.cases()}
    task = cases["create_task_1"]
    actions = task.metadata["evaluation_criteria"]["actions"]
    assert actions[0]["name"] == "create_task"
    assert actions[0]["arguments"]["title"] == "Important Meeting"


def test_missing_domain_raises_clear_error():
    import pytest
    with pytest.raises(FileNotFoundError):
        Tau2Benchmark(domain="does_not_exist")
