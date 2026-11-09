"""Launch independent frozen tasks on explicitly assigned local GPUs."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protein-gpus", nargs="+", type=int, required=True)
    parser.add_argument("--rna-gpus", nargs="*", type=int, default=[])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rebalance-protein", action="store_true",
                        help="After RNA40 completes, stop only this experiment's old protein workers and resume on the supplied GPU pool")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    runner = root / "grabspark/run_raw_probe_experiment.py"
    output = args.output.resolve()
    if not (output / "models/fig2/Group3/complete.json").exists():
        raise ValueError("Fig2 must complete before Stage B")
    logdir = output / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    if args.rebalance_protein:
        if args.rna_gpus or len(list((output / "models/rna40").glob("*/complete.json"))) != 40:
            raise ValueError("Rebalance requires all RNA40 tasks complete and an empty RNA GPU list")
        old_commands = json.loads((logdir / "worker_commands.json").read_text())
        (logdir / "initial_worker_commands.json").write_text(json.dumps(old_commands, indent=2)+"\n")
        stopped = []
        for entry in old_commands:
            if entry["suite"] != "protein60":
                continue
            pid = int(entry["pid"])
            cmd_path = Path(f"/proc/{pid}/cmdline")
            if not cmd_path.exists():
                continue
            try:
                actual = [part.decode() for part in cmd_path.read_bytes().split(b"\0") if part]
            except FileNotFoundError:
                continue
            # Never signal a PID unless it is still our exact launched command.
            if actual != entry["argv"] or str(output) not in actual or str(runner) not in actual:
                raise ValueError(f"PID {pid} no longer matches the launched worker")
            try:
                os.kill(pid, signal.SIGTERM)
                stopped.append(pid)
            except ProcessLookupError:
                pass
        deadline = time.monotonic()+30
        def still_running(pid: int) -> bool:
            try:
                return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
            except FileNotFoundError:
                return False
        while any(still_running(pid) for pid in stopped):
            if time.monotonic() > deadline:
                raise RuntimeError("Old workers did not exit; no replacement workers launched")
            time.sleep(.2)
        (logdir / "rebalance.json").write_text(json.dumps({"reason": "RNA40 finished; use all eight GPUs for remaining protein tasks",
                                                          "stopped_owned_worker_pids": stopped,
                                                          "resume_policy": "Skip complete tasks; reuse authenticated complete caches; re-extract partial caches"}, indent=2)+"\n")
        print(f"REBALANCE stopped own worker PIDs: {stopped}", flush=True)
    environment = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        environment[key] = "4"
    environment["TRITON_CACHE_DIR"] = str(output / "runtime/triton")
    workers = []
    commands = []
    for suite, devices in (("protein60", args.protein_gpus), ("rna40", args.rna_gpus)):
        for index, device in enumerate(devices):
            command = [sys.executable, "-u", str(runner), "--stage", "run", "--suite", suite,
                       "--output", str(output), "--device", f"cuda:{device}",
                       "--workers", str(len(devices)), "--worker-index", str(index),
                       "--batch-size", str(args.batch_size)]
            handle = (logdir / f"{suite}_worker{index}_gpu{device}.log").open("a")
            handle.write("COMMAND: " + " ".join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, env=environment)
            workers.append((process, handle, suite, index))
            commands.append({"argv": command, "pid": process.pid, "suite": suite, "worker": index})
            print(f"START {suite}/{index} GPU {device} PID {process.pid}", flush=True)
    (logdir / "worker_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    failures, exits = [], []
    for process, handle, suite, index in workers:
        code = process.wait()
        handle.close()
        print(f"EXIT {suite}/{index}: {code}", flush=True)
        exits.append({"suite": suite, "worker": index, "exit_code": code})
        if code:
            failures.append({"suite": suite, "worker": index, "exit_code": code})
    (logdir / "worker_exit_status.json").write_text(json.dumps(exits, indent=2) + "\n")
    aggregate = [sys.executable, "-u", str(runner), "--stage", "aggregate", "--output", str(output)]
    subprocess.run(aggregate, cwd=root, env=environment, check=True)
    if failures:
        raise SystemExit("Some workers failed; partial results aggregated, inspect logs")


if __name__ == "__main__":
    main()
