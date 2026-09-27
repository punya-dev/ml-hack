"""
SageMaker Blocking Pipeline Manager.
Orchestrates S3 data sync, code packaging, SageMaker job creation, real-time CloudWatch streaming,
and artifact retrieval for Train and Test Candidate Blocking.
"""
import os
import sys
import time
import glob
import argparse
from typing import Optional, List
import boto3
from botocore.exceptions import ClientError

REGION = "ap-southeast-2"
PROFILE = "ml-hackathon"
BUCKET = "sagemaker-ap-southeast-2-808939435584"
ROLE_ARN = "arn:aws:iam::808939435584:role/service-role/AmazonSageMaker-ExecutionRole-20260925T223430"
IMAGE_URI = "783357654285.dkr.ecr.ap-southeast-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

def get_session():
    try:
        return boto3.Session(profile_name=PROFILE, region_name=REGION)
    except Exception:
        return boto3.Session(region_name=REGION)

from boto3.s3.transfer import TransferConfig

TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=16 * 1024 * 1024,
    max_concurrency=20,
    multipart_chunksize=16 * 1024 * 1024,
    use_threads=True,
)

def upload_file_if_needed(s3_client, local_path: str, bucket: str, s3_key: str):
    local_size = os.path.getsize(local_path)
    try:
        head = s3_client.head_object(Bucket=bucket, Key=s3_key)
        remote_size = head.get("ContentLength", 0)
        if remote_size == local_size:
            print(f"  [S3 Cache] {s3_key} ({local_size / (1024*1024):.1f} MB) already up to date.")
            return
    except ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise

    print(f"  [Uploading] {local_path} -> s3://{bucket}/{s3_key} ({local_size / (1024*1024):.1f} MB)...")
    t0 = time.time()
    s3_client.upload_file(local_path, bucket, s3_key, Config=TRANSFER_CONFIG)
    elapsed = time.time() - t0
    rate = (local_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
    print(f"  [Uploaded] in {elapsed:.1f}s ({rate:.2f} MB/s).")

def upload_code(session):
    s3 = session.client("s3")
    print("\n" + "=" * 80)
    print("PACKAGING & UPLOADING CODE TO S3")
    print("=" * 80)
    
    # 1. Upload scripts/run_sagemaker_blocking_worker.py as run_worker.py
    worker_script = os.path.abspath("scripts/run_sagemaker_blocking_worker.py")
    upload_file_if_needed(s3, worker_script, BUCKET, "blocking_job/code/run_worker.py")

    # 2. Upload src/ directory
    src_files = glob.glob("code/business_entity_resolution/src/*.py") or glob.glob("src/*.py")
    for src_file in src_files:
        fname = os.path.basename(src_file)
        upload_file_if_needed(s3, src_file, BUCKET, f"blocking_job/code/src/{fname}")

    # 3. Upload configs/ directory
    cfg_files = glob.glob("code/business_entity_resolution/configs/*.json") or glob.glob("configs/*.json")
    for cfg_file in cfg_files:
        fname = os.path.basename(cfg_file)
        upload_file_if_needed(s3, cfg_file, BUCKET, f"blocking_job/code/configs/{fname}")

    print("Code upload complete.")

def upload_data(session, mode: str):
    s3 = session.client("s3")
    print("\n" + "=" * 80)
    print(f"UPLOADING {mode.upper()} DATASETS TO S3")
    print("=" * 80)
    
    if mode in ["train", "both"]:
        train_files = [
            "dataset/train/train_source1.tsv",
            "dataset/train/train_source2.tsv",
            "dataset/train/train_source3.tsv",
            "dataset/train/train_ground_truth.tsv",
        ]
        for fpath in train_files:
            if os.path.exists(fpath):
                fname = os.path.basename(fpath)
                upload_file_if_needed(s3, fpath, BUCKET, f"blocking_job/data/train/{fname}")
            else:
                print(f"  [Warning] Missing {fpath}")

    if mode in ["test", "both"]:
        test_files = [
            "dataset/test/test_source1.tsv",
            "dataset/test/test_source2.tsv",
            "dataset/test/test_source3.tsv",
        ]
        for fpath in test_files:
            if os.path.exists(fpath):
                fname = os.path.basename(fpath)
                upload_file_if_needed(s3, fpath, BUCKET, f"blocking_job/data/test/{fname}")
            else:
                print(f"  [Warning] Missing {fpath}")

    print(f"Data upload for {mode} complete.")

def launch_blocking_job(session, mode: str, instance_type: str = "ml.t3.xlarge") -> str:
    sm = session.client("sagemaker")
    timestamp = int(time.time())
    job_name = f"blocking-{mode}-{timestamp}"
    
    print("\n" + "=" * 80)
    print(f"LAUNCHING SAGEMAKER PROCESSING JOB: {job_name}")
    print(f"  Mode:          {mode}")
    print(f"  Instance Type: {instance_type}")
    print(f"  Role:          {ROLE_ARN}")
    print(f"  Image:         {IMAGE_URI}")
    print("=" * 80)

    resp = sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": 50,
            }
        },
        AppSpecification={
            "ImageUri": IMAGE_URI,
            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/input/code/run_worker.py",
                "--mode", mode,
            ],
        },
        RoleArn=ROLE_ARN,
        ProcessingInputs=[
            {
                "InputName": "code",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/blocking_job/code",
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "data",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/blocking_job/data/{mode}",
                    "LocalPath": "/opt/ml/processing/input/data",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "output",
                    "AppManaged": False,
                    "S3Output": {
                        "S3Uri": f"s3://{BUCKET}/blocking_job/output/{job_name}",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
    )

    print(f"Job launched successfully. ARN: {resp['ProcessingJobArn']}")
    return job_name

def stream_logs(session, job_name: str):
    sm = session.client("sagemaker")
    logs = session.client("logs")
    log_group = "/aws/sagemaker/ProcessingJobs"

    print(f"\nMonitoring job {job_name} (tailing CloudWatch logs)...")
    seen_token = None
    seen_event_ids = set()

    while True:
        desc = sm.describe_processing_job(ProcessingJobName=job_name)
        status = desc["ProcessingJobStatus"]

        # Fetch log streams
        try:
            streams = logs.describe_log_streams(
                logGroupName=log_group,
                logStreamNamePrefix=job_name,
                orderBy="LogStreamName",
            ).get("logStreams", [])

            for s in streams:
                s_name = s["logStreamName"]
                kwargs = {
                    "logGroupName": log_group,
                    "logStreamName": s_name,
                    "startFromHead": True,
                }
                if seen_token:
                    kwargs["nextToken"] = seen_token
                events_resp = logs.get_log_events(**kwargs)
                for ev in events_resp.get("events", []):
                    if ev["eventId"] not in seen_event_ids:
                        seen_event_ids.add(ev["eventId"])
                        print(f"[{job_name}] {ev['message'].rstrip()}")
                seen_token = events_resp.get("nextForwardToken")
        except Exception:
            pass

        if status in ["Completed", "Failed", "Stopped"]:
            print(f"\nJob finished with status: {status}")
            if status == "Failed":
                print(f"FailureReason: {desc.get('FailureReason')}")
            break

        time.sleep(15)

def download_outputs(session, job_name: str, mode: str):
    s3 = session.client("s3")
    print("\n" + "=" * 80)
    print(f"DOWNLOADING {mode.upper()} OUTPUTS FROM S3 FOR JOB: {job_name}")
    print("=" * 80)
    
    os.makedirs("output", exist_ok=True)
    prefix = f"blocking_job/output/{job_name}/"
    paginator = s3.get_paginator("list_objects_v2")
    
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = os.path.basename(key)
            if not fname:
                continue
            dest_path = os.path.join("output", fname)
            size = obj["Size"]
            print(f"  Downloading s3://{BUCKET}/{key} ({size / (1024*1024):.1f} MB) -> {dest_path}...")
            t0 = time.time()
            s3.download_file(BUCKET, key, dest_path)
            print(f"  Downloaded in {time.time() - t0:.1f}s.")

def main():
    parser = argparse.ArgumentParser(description="SageMaker Candidate Blocking Orchestration")
    parser.add_argument("--action", choices=["upload-code", "upload-data", "launch", "monitor", "download", "all"], required=True)
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--job-name", default=None, help="Job name for monitor or download")
    parser.add_argument("--instance-type", default="ml.t3.xlarge")
    args = parser.parse_args()

    session = get_session()

    if args.action in ["upload-code", "all"]:
        upload_code(session)

    if args.action in ["upload-data", "all"]:
        upload_data(session, args.mode)

    if args.action in ["launch", "all"]:
        modes_to_launch = ["train", "test"] if args.mode == "both" else [args.mode]
        job_names = []
        for m in modes_to_launch:
            jn = launch_blocking_job(session, m, instance_type=args.instance_type)
            job_names.append((jn, m))

        if args.action == "all":
            for jn, m in job_names:
                stream_logs(session, jn)
                download_outputs(session, jn, m)

    if args.action == "monitor" and args.job_name:
        stream_logs(session, args.job_name)

    if args.action == "download" and args.job_name:
        download_outputs(session, args.job_name, args.mode)

if __name__ == "__main__":
    main()
