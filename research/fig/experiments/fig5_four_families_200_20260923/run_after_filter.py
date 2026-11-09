"""Wait for successful cohort freeze, then execute all frozen evaluations."""
from pathlib import Path
import json
import subprocess
import sys
import time
OUT=Path(__file__).resolve().parent
start=time.monotonic()
while not (OUT/'selection_manifest.json').exists():
    if (OUT/'eligible_cluster_counts.json').exists():
        counts=json.loads((OUT/'eligible_cluster_counts.json').read_text())
        if any(n<200 for n in counts.values()):
            raise SystemExit(f'Need additional candidates before inference: {counts}')
    if time.monotonic()-start>3600:
        raise SystemExit('No completed cohort freeze after one hour; inspect filter.log')
    time.sleep(5)
subprocess.run([sys.executable,str(OUT/'run_inference.py')],check=True)
subprocess.run([sys.executable,str(OUT/'write_report.py')],check=True)
