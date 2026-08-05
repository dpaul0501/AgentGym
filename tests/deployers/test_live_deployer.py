import pytest

from agentgym.deployers.live_deployer import LiveDeployer


def test_release_carries_traffic_pct_and_starts_with_no_outcomes():
    deployer = LiveDeployer(seed=0)
    release = deployer.release(variant="candidate-v2", traffic_pct=0.1)
    assert release.variant == "candidate-v2"
    assert release.live.traffic_pct == 0.1
    assert release.live.baseline_outcomes == []
    assert release.live.candidate_outcomes == []


def test_route_is_deterministic_for_the_same_request_id():
    deployer = LiveDeployer(seed=0)
    release = deployer.release(variant="candidate", traffic_pct=0.5)
    first = deployer.route(release, request_id="req-42", baseline_variant="baseline")
    second = deployer.route(release, request_id="req-42", baseline_variant="baseline")
    assert first == second


def test_route_splits_traffic_roughly_by_traffic_pct():
    deployer = LiveDeployer(seed=0)
    release = deployer.release(variant="candidate", traffic_pct=0.3)
    routed = [deployer.route(release, request_id=f"req-{i}", baseline_variant="baseline") for i in range(2000)]
    candidate_fraction = routed.count("candidate") / len(routed)
    assert 0.24 < candidate_fraction < 0.36


def test_record_outcome_attributes_to_correct_bucket():
    deployer = LiveDeployer(seed=0)
    release = deployer.release(variant="candidate", traffic_pct=0.5)
    deployer.record_outcome(release, used_variant="candidate", score=1.0)
    deployer.record_outcome(release, used_variant="baseline", score=0.0)
    assert release.live.candidate_outcomes == [1.0]
    assert release.live.baseline_outcomes == [0.0]


def test_ab_compare_refuses_when_not_enough_real_outcomes_yet():
    deployer = LiveDeployer(seed=0, min_outcomes_for_ab=30)
    release = deployer.release(variant="candidate", traffic_pct=0.5)
    for _ in range(5):
        deployer.record_outcome(release, used_variant="candidate", score=1.0)
        deployer.record_outcome(release, used_variant="baseline", score=0.0)

    with pytest.raises(ValueError):
        deployer.ab_compare(baseline="baseline", candidate="candidate", release=release)


def test_ab_compare_reports_a_real_win_from_real_accumulated_outcomes():
    deployer = LiveDeployer(seed=0, min_outcomes_for_ab=20)
    release = deployer.release(variant="candidate", traffic_pct=0.5)
    for _ in range(20):
        deployer.record_outcome(release, used_variant="candidate", score=1.0)
        deployer.record_outcome(release, used_variant="baseline", score=0.0)

    result = deployer.ab_compare(baseline="baseline", candidate="candidate", release=release)

    assert result.delta > 0
    assert result.candidate_wins is True


def test_rollback_stops_routing_new_traffic_to_the_candidate():
    deployer = LiveDeployer(seed=0)
    release = deployer.release(variant="candidate", traffic_pct=1.0)
    deployer.rollback(release)
    routed = deployer.route(release, request_id="req-1", baseline_variant="baseline")
    assert routed == "baseline"
