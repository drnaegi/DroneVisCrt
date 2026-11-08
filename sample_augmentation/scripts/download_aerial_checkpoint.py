"""Resumable parallel range download of the pinned official MASt3R weights."""
import concurrent.futures
import hashlib
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = 'https://huggingface.co/kvuong2711/checkpoint-aerial-mast3r/resolve/0cf76e1da5c9b2bb7fb9d17cce03e0e6582c034e/model.safetensors'
SIZE = 2754661648
SHA256 = '093c3b4b3ad8294673774aae106309d5f6a1a975976bae574075fb7c5f94a67d'


def main():
    folder = ROOT / 'weights/aerial_mast3r'
    destination = folder / 'model.safetensors'
    if destination.exists():
        if hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest() == SHA256:
            print('Already verified', flush=True)
            return
        raise RuntimeError('Existing checkpoint does not match official hash')
    cache = folder / 'range_cache'
    cache.mkdir(exist_ok=True)
    chunk = 16 * 1024 * 1024
    jobs = [(a, min(a + chunk, SIZE)) for a in range(0, SIZE, chunk)]

    def fetch(job):
        a, b = job
        path = cache / f'{a}_{b}.bin'
        if path.exists() and path.stat().st_size == b - a:
            return b - a
        for attempt in range(5):
            try:
                request = urllib.request.Request(URL + f'?range_start={a}', headers={'Range': f'bytes={a}-{b-1}'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {a}-{b-1}/{SIZE}':
                        raise RuntimeError('Unexpected range response')
                    data = response.read()
                if len(data) != b - a:
                    raise RuntimeError('Truncated download')
                temp = path.with_suffix('.tmp')
                temp.write_bytes(data)
                temp.replace(path)
                return len(data)
            except Exception as error:
                print('Retry', a, attempt, type(error).__name__, flush=True)
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    start = time.monotonic()
    nbytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            nbytes += future.result()
            if i % 8 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} chunks, {nbytes/2**30:.2f} GiB, {time.monotonic()-start:.0f}s', flush=True)
    temporary = destination.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        for a, b in jobs:
            stream.write((cache / f'{a}_{b}.bin').read_bytes())
    digest = hashlib.file_digest(temporary.open('rb'), 'sha256').hexdigest()
    if digest != SHA256:
        raise RuntimeError(f'SHA256 mismatch: {digest}')
    temporary.replace(destination)
    print('Verified SHA256:', digest, flush=True)


if __name__ == '__main__':
    main()
