"""Unit tests for the pure, no-AWS-call pieces of AWSGPUJobRunner. Anything that actually calls
boto3 (launch/wait_and_fetch, ensure_bucket/ensure_iam_role) is real, billable AWS infrastructure
and is deliberately NOT exercised here — only via scripts/provision_gpu.py, run manually."""

import json

from agentgym.providers.aws_gpu_runner import build_user_data, bucket_policy


def test_build_user_data_references_the_right_s3_prefix_and_bucket():
    script = build_user_data(bucket="agentgym-gpu-runs-12345", run_id="abc123")
    assert "s3://agentgym-gpu-runs-12345/runs/abc123/config.json" in script
    assert "s3://agentgym-gpu-runs-12345/runs/abc123/data.json" in script
    assert "s3://agentgym-gpu-runs-12345/runs/abc123/adapter_out" in script


def test_build_user_data_self_terminates_after_uploading_results():
    script = build_user_data(bucket="b", run_id="r1")
    assert "touch DONE" in script
    assert "shutdown -h now" in script


def test_bucket_policy_scopes_to_exactly_one_bucket():
    policy = json.loads(bucket_policy("agentgym-gpu-runs-12345"))
    resources = policy["Statement"][0]["Resource"]
    assert resources == [
        "arn:aws:s3:::agentgym-gpu-runs-12345",
        "arn:aws:s3:::agentgym-gpu-runs-12345/*",
    ]
