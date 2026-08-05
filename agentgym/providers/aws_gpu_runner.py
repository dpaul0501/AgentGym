"""AWSGPUJobRunner: the ONE real implementation of optimizers.unsloth_lora.CloudGPUJobRunner that
actually spends money. It launches a real EC2 GPU instance, runs agentgym/optimizers/
_unsloth_train.py on it via user-data, and fetches the resulting LoRA adapter back from S3.

This module is never imported by agentgym's core package or by any pytest-collected test —
constructing AWSGPUJobRunner and calling .launch() are real, billable AWS actions. The only
intended caller is scripts/provision_gpu.py, run manually after the type/cost/duration
confirmation checkpoint.

Design, chosen after reading the real state of this AWS account rather than guessing: no SSH key
pair and no inbound security-group rule are used — the instance runs its job entirely from
user-data (root, at boot), reports completion via an S3 marker object, and self-terminates. Uses
the account's existing default VPC/default security group (outbound-only is sufficient) rather
than the pre-existing "gordon-*" VPC/security groups, which belong to unrelated infrastructure and
are left untouched. A dedicated S3 bucket and a minimal IAM role, scoped to only that bucket, are
created for this purpose (ensure_bucket/ensure_iam_role) rather than reusing any pre-existing
bucket or role in the account.
"""

from __future__ import annotations

import json
import textwrap
import time
import uuid
from pathlib import Path

import boto3

TRAIN_SCRIPT_PATH = Path(__file__).parent.parent / "optimizers" / "_unsloth_train.py"

TRUST_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}],
})


def bucket_policy(bucket: str) -> str:
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
        }],
    })


def build_user_data(bucket: str, run_id: str) -> str:
    """The script that runs as root at boot on the training instance. Pure function of
    (bucket, run_id) — no AWS calls, so this is unit-testable without touching real infra."""
    prefix = f"runs/{run_id}"
    return textwrap.dedent(f"""\
        #!/bin/bash
        set -ex
        mkdir -p /tmp/agentgym_job
        cd /tmp/agentgym_job
        aws s3 cp s3://{bucket}/{prefix}/config.json config.json
        aws s3 cp s3://{bucket}/{prefix}/data.json data.json
        aws s3 cp s3://{bucket}/{prefix}/_unsloth_train.py _unsloth_train.py
        source /opt/pytorch/bin/activate || true
        pip install -q unsloth trl peft
        python _unsloth_train.py
        aws s3 cp adapter_out s3://{bucket}/{prefix}/adapter_out --recursive
        aws s3 cp metrics.json s3://{bucket}/{prefix}/metrics.json
        touch DONE
        aws s3 cp DONE s3://{bucket}/{prefix}/DONE
        shutdown -h now
        """)


class AWSGPUJobRunner:
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        instance_type: str = "g6.xlarge",
        ami_id: str | None = None,
        instance_profile_name: str = "agentgym-gpu-runner",
        volume_size_gb: int = 100,
    ):
        self.bucket = bucket
        self.region = region
        self.instance_type = instance_type
        self.instance_profile_name = instance_profile_name
        self.volume_size_gb = volume_size_gb
        self.s3 = boto3.client("s3", region_name=region)
        self.ec2 = boto3.client("ec2", region_name=region)
        self.iam = boto3.client("iam")
        self.ami_id = ami_id or self._latest_dl_ami()
        self._instance_ids: dict[str, str] = {}

    def _latest_dl_ami(self) -> str:
        ssm = boto3.client("ssm", region_name=self.region)
        param = (
            "/aws/service/deeplearning/ami/x86_64/"
            "oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id"
        )
        return ssm.get_parameter(Name=param)["Parameter"]["Value"]

    def ensure_bucket(self) -> None:
        existing = {b["Name"] for b in self.s3.list_buckets()["Buckets"]}
        if self.bucket not in existing:
            if self.region == "us-east-1":
                self.s3.create_bucket(Bucket=self.bucket)
            else:
                self.s3.create_bucket(
                    Bucket=self.bucket,
                    CreateBucketConfiguration={"LocationConstraint": self.region},
                )

    def ensure_iam_role(self) -> None:
        try:
            self.iam.get_role(RoleName=self.instance_profile_name)
        except self.iam.exceptions.NoSuchEntityException:
            self.iam.create_role(
                RoleName=self.instance_profile_name, AssumeRolePolicyDocument=TRUST_POLICY,
            )
        self.iam.put_role_policy(
            RoleName=self.instance_profile_name, PolicyName="agentgym-s3-access",
            PolicyDocument=bucket_policy(self.bucket),
        )
        try:
            self.iam.get_instance_profile(InstanceProfileName=self.instance_profile_name)
        except self.iam.exceptions.NoSuchEntityException:
            self.iam.create_instance_profile(InstanceProfileName=self.instance_profile_name)
            self.iam.add_role_to_instance_profile(
                InstanceProfileName=self.instance_profile_name, RoleName=self.instance_profile_name,
            )
            time.sleep(10)  # IAM->EC2 propagation delay before the profile is usable at launch

    def _default_subnet(self) -> str:
        vpcs = self.ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
        default_vpc_id = vpcs[0]["VpcId"]
        subnets = self.ec2.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [default_vpc_id]}]
        )["Subnets"]
        return subnets[0]["SubnetId"]

    def launch(self, script_path: str, data: list[dict], base_model: str, lora_rank: int) -> str:
        self.ensure_bucket()
        self.ensure_iam_role()

        run_id = uuid.uuid4().hex[:12]
        prefix = f"runs/{run_id}"
        self.s3.put_object(Bucket=self.bucket, Key=f"{prefix}/data.json", Body=json.dumps(data))
        self.s3.put_object(
            Bucket=self.bucket, Key=f"{prefix}/config.json",
            Body=json.dumps({"base_model": base_model, "lora_rank": lora_rank}),
        )
        self.s3.upload_file(str(TRAIN_SCRIPT_PATH), self.bucket, f"{prefix}/_unsloth_train.py")

        response = self.ec2.run_instances(
            ImageId=self.ami_id,
            InstanceType=self.instance_type,
            MinCount=1,
            MaxCount=1,
            SubnetId=self._default_subnet(),
            IamInstanceProfile={"Name": self.instance_profile_name},
            InstanceInitiatedShutdownBehavior="terminate",
            BlockDeviceMappings=[{
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": self.volume_size_gb, "VolumeType": "gp3"},
            }],
            UserData=build_user_data(self.bucket, run_id),
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": f"agentgym-lora-{run_id}"}],
            }],
        )
        self._instance_ids[run_id] = response["Instances"][0]["InstanceId"]
        return run_id

    def wait_and_fetch(self, run_id: str, poll_interval: int = 30, timeout: int = 3600) -> dict:
        prefix = f"runs/{run_id}"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.s3.head_object(Bucket=self.bucket, Key=f"{prefix}/DONE")
                break
            except self.s3.exceptions.ClientError:
                time.sleep(poll_interval)
        else:
            raise TimeoutError(f"run {run_id} did not finish within {timeout}s")

        local_dir = Path(f"/tmp/agentgym_runs/{run_id}")
        local_dir.mkdir(parents=True, exist_ok=True)
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{prefix}/adapter_out/"):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(f"{prefix}/adapter_out/"):]
                dest = local_dir / "adapter_out" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(self.bucket, obj["Key"], str(dest))
        metrics_dest = local_dir / "metrics.json"
        self.s3.download_file(self.bucket, f"{prefix}/metrics.json", str(metrics_dest))
        metrics = json.loads(metrics_dest.read_text())

        instance_id = self._instance_ids.get(run_id)
        if instance_id:
            self.ec2.terminate_instances(InstanceIds=[instance_id])

        return {"adapter_path": str(local_dir / "adapter_out"), "metrics": metrics}
