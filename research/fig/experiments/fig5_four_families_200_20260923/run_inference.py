"""Four independent CPU workers; frozen cohort/readouts, resumable per protein."""
from pathlib import Path
import concurrent.futures
import subprocess
import sys
OUT=Path(__file__).resolve().parent

def worker(family):
    with (OUT/f'inference_{family}.log').open('a') as log:
        subprocess.run([sys.executable,str(OUT/'infer_new_sequences.py'),'--family',family],stdout=log,stderr=subprocess.STDOUT,check=True)
    return family

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for family in pool.map(worker,['PF00004','PF00005','PF00069','PF07714']):
            print(f'{family} completed',flush=True)
    subprocess.run([sys.executable,str(OUT/'evaluate_external.py')],check=True)
if __name__=='__main__':
    main()
