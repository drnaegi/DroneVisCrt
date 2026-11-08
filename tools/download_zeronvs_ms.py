import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'checkpoints/zeronvs-megascenes'
URL = 'https://s3.us-west-2.amazonaws.com/megascenes/nvs_checkpoints/zeronvs_finetune/iter_90000/model.safetensors'
SIZE = 8583857168
ETAG = 'cfd91b18bbaba0026594e90359a60c37-164'
CHUNK_SIZE = 2 * 1024 * 1024


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    partial = DIRECTORY / 'model.safetensors.part'
    target = DIRECTORY / 'model.safetensors'
    state_path = DIRECTORY / 'download_state.json'
    if target.exists():
        print('Finalized checkpoint already exists:', target)
        return
    completed = set()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['etag'] != ETAG or not partial.exists():
            raise ValueError('Resume metadata does not match')
        previous_chunk_size = state.get('chunk_size', 8 * 1024 * 1024)
        if previous_chunk_size % CHUNK_SIZE:
            raise ValueError('Cannot resume with incompatible chunk size')
        completed = {offset for start in state['completed']
                     for offset in range(start, min(start + previous_chunk_size, SIZE), CHUNK_SIZE)}
    descriptor = os.open(str(partial), os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(descriptor, SIZE)
    ranges = [(start, min(start + CHUNK_SIZE, SIZE) - 1)
              for start in range(0, SIZE, CHUNK_SIZE) if start not in completed]
    started = time.monotonic()
    initial_count = len(completed)

    def fetch(bounds):
        start, end = bounds
        for attempt in range(5):
            try:
                with requests.get(URL, headers={'Range': 'bytes=%d-%d' % (start, end),
                                                'If-Match': '"' + ETAG + '"'},
                                  stream=True, timeout=(15, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != 'bytes %d-%d/%d' % (start, end, SIZE):
                        raise ValueError('Invalid range response')
                    position = start
                    for chunk in response.iter_content(256 * 1024):
                        if position + len(chunk) > end + 1:
                            raise ValueError('Oversized range')
                        written = os.pwrite(descriptor, chunk, position)
                        if written != len(chunk):
                            raise OSError('Short write')
                        position += written
                    if position != end + 1:
                        raise ValueError('Incomplete range')
                return start
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            futures = [pool.submit(fetch, bounds) for bounds in ranges]
            for future in concurrent.futures.as_completed(futures):
                completed.add(future.result())
                temporary = state_path.with_suffix('.tmp')
                temporary.write_text(json.dumps({'url': URL, 'size': SIZE, 'etag': ETAG, 'chunk_size': CHUNK_SIZE,
                                                 'completed': sorted(completed)}) + '\n')
                temporary.replace(state_path)
                count = len(completed)
                if (count - initial_count) % 8 == 0:
                    elapsed = time.monotonic() - started
                    print('%.1f%% verified ranges; %.2f MiB/s' % (
                        min(count * CHUNK_SIZE / SIZE * 100, 100),
                        (count - initial_count) * CHUNK_SIZE / elapsed / 1024 ** 2), flush=True)
    finally:
        os.close(descriptor)
    multipart = []
    sha256 = hashlib.sha256()
    with partial.open('rb') as stream:
        for chunk in iter(lambda: stream.read(50 * 1024 * 1024), b''):
            multipart.append(hashlib.md5(chunk).digest())
            sha256.update(chunk)
    actual_etag = hashlib.md5(b''.join(multipart)).hexdigest() + '-' + str(len(multipart))
    if actual_etag != ETAG:
        raise ValueError('S3 multipart checksum mismatch: ' + actual_etag)
    partial.rename(target)
    (DIRECTORY / 'model.safetensors.sha256').write_text(sha256.hexdigest() + '  model.safetensors\n')
    print('Download complete; official S3 multipart ETag verified:', actual_etag, flush=True)


if __name__ == '__main__':
    main()
