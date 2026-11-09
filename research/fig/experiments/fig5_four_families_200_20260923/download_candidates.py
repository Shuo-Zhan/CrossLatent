"""Cache paginated public InterPro records; never submits local sequences."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request

OUT = Path(__file__).resolve().parent
FAMILIES = ['PF00004', 'PF00005', 'PF00069', 'PF07714']

def download(family: str, pages: int) -> dict:
    url = f'https://www.ebi.ac.uk/interpro/api/protein/uniprot/entry/pfam/{family}/?page_size=100&extra_fields=sequence'
    manifest = []
    for page in range(1, pages + 1):
        path = OUT / 'raw' / f'{family}_page{page:04d}.json'
        if path.exists():
            raw = path.read_bytes()
            payload = json.loads(raw)
        else:
            for attempt in range(6):
                try:
                    request = urllib.request.Request(url, headers={'User-Agent': 'CrossLatent-research/1.0', 'Accept': 'application/json'})
                    with urllib.request.urlopen(request, timeout=180) as response:
                        raw = response.read()
                    payload = json.loads(raw)
                    assert isinstance(payload['results'], list)
                    temp = path.with_suffix('.partial')
                    temp.write_bytes(raw)
                    temp.replace(path)
                    break
                except Exception as exc:
                    print(f'{family} page {page} attempt {attempt+1}: {exc}', flush=True)
                    if attempt == 5:
                        raise
                    time.sleep(min(5 * (attempt + 1), 30))
        manifest.append({'page': page, 'url': url, 'sha256': hashlib.sha256(raw).hexdigest(), 'n_records': len(payload['results'])})
        (OUT / f'acquisition_{family}.json').write_text(json.dumps(manifest, indent=2))
        print(f'{family} page {page}: {len(payload["results"])} records cached', flush=True)
        url = payload.get('next')
        if not url:
            break
    return {'family': family, 'records': sum(p['n_records'] for p in manifest)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages', type=int, default=20)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda family: download(family, args.pages), args.families):
            print(result, flush=True)

if __name__ == '__main__':
    main()
