"""Run run.sh on a fresh RunPod GPU pod, fetch results/, terminate the pod.

    .venv/bin/python pod.py "NVIDIA GeForce RTX 4090"
    .venv/bin/python pod.py --list          # GPU type ids and prices
    .venv/bin/python pod.py "NVIDIA GeForce RTX 5090" --blackwell   # CUDA 12.8 image

API key: Keychain item `runpod-api-key`. SSH key: ~/.ssh/julien-runpod.
The pod is terminated in a finally block, whatever happens.
"""

import subprocess
import sys
import time
from pathlib import Path

import runpod

HERE = Path(__file__).resolve().parent
KEY = Path.home() / ".ssh" / "julien-runpod"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# CUDA 12.4 has no sm_120 cuBLAS kernels; Blackwell needs 12.8 or later.
IMAGE_BLACKWELL = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
SSH_OPTS = ["-i", str(KEY), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", "-o", "LogLevel=ERROR"]


def api_key():
    return subprocess.run(["security", "find-generic-password", "-s", "runpod-api-key", "-w"],
                          capture_output=True, text=True, check=True).stdout.strip()


def ssh_endpoint(pod_id, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = runpod.get_pod(pod_id) or {}
        for p in ((pod.get("runtime") or {}).get("ports") or []):
            if p.get("privatePort") == 22 and p.get("isIpPublic"):
                return p["ip"], p["publicPort"]
        time.sleep(10)
    raise TimeoutError(f"pod {pod_id}: no public SSH port after {timeout} s")


def run(gpu_type, image):
    pod = runpod.create_pod(
        name="geforce-cap", image_name=image, gpu_type_id=gpu_type, cloud_type="ALL",
        gpu_count=1, volume_in_gb=0, container_disk_in_gb=20, ports="22/tcp",
        support_public_ip=True, env={"PUBLIC_KEY": KEY.with_suffix(".pub").read_text().strip()},
    )
    pod_id = pod["id"]
    print(f"pod {pod_id} created on {gpu_type}", flush=True)
    try:
        ip, port = ssh_endpoint(pod_id)
        target = f"root@{ip}"
        port_opt = ["-p", str(port)]
        for _ in range(30):  # sshd starts a little after the port is mapped
            if subprocess.run(["ssh", *SSH_OPTS, *port_opt, target, "true"]).returncode == 0:
                break
            time.sleep(10)
        subprocess.run(["scp", *SSH_OPTS, "-P", str(port), str(HERE / "run.sh"), f"{target}:/root/run.sh"],
                       check=True)
        subprocess.run(["ssh", *SSH_OPTS, *port_opt, target, "WORKDIR=/root/geforce-cap bash /root/run.sh"],
                       check=True)
        (HERE / "results").mkdir(exist_ok=True)
        subprocess.run(["scp", *SSH_OPTS, "-r", "-P", str(port), f"{target}:/root/geforce-cap/results/*",
                        str(HERE / "results")], check=True)
    finally:
        runpod.terminate_pod(pod_id)
        print(f"pod {pod_id} terminated", flush=True)


def main():
    runpod.api_key = api_key()
    if sys.argv[1:] == ["--list"]:
        for g in runpod.get_gpus():
            print(g["id"], "|", g.get("memoryInGb"), "GB")
        return
    args = [a for a in sys.argv[1:] if a != "--blackwell"]
    if len(args) != 1:
        sys.exit(__doc__)
    run(args[0], IMAGE_BLACKWELL if "--blackwell" in sys.argv else IMAGE)


if __name__ == "__main__":
    main()
