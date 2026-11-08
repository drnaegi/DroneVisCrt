import concurrent.futures
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import requests

from download_zeronvs_ms import DIRECTORY, ETAG, SIZE, URL


def main():
    partial = DIRECTORY / 'model.safetensors.part'
    state_path = DIRECTORY / 'download_state.json'
    state = json.loads(state_path.read_text())
    chunk_size = state.get('chunk_size', 8 * 1024 * 1024)
    completed = set(state['completed'])
    with partial.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        header_bytes = stream.read(header_size)
    header = json.loads(header_bytes)
    data_start = 8 + header_size
    selected = {name: value for name, value in header.items()
                if name != '__metadata__' and not name.startswith('model_ema.')}
    needed = set()
    for value in selected.values():
        start, end = [data_start + offset for offset in value['data_offsets']]
        needed.update(range(start // chunk_size * chunk_size, end, chunk_size))
    missing = sorted(needed - completed)
    print('Inference tensors:', len(selected), '; missing bytes:',
          sum(min(chunk_size, SIZE - start) for start in missing), flush=True)
    descriptor = os.open(str(partial), os.O_RDWR)

    def fetch(start):
        end = min(start + chunk_size, SIZE) - 1
        for attempt in range(5):
            try:
                response = requests.get(URL, headers={'Range': 'bytes=%d-%d' % (start, end),
                                                      'If-Match': '"' + ETAG + '"'}, timeout=(15, 90))
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != 'bytes %d-%d/%d' % (start, end, SIZE):
                    raise ValueError('Incorrect HTTP range')
                if response.headers.get('ETag', '').strip('"') != ETAG or len(response.content) != end - start + 1:
                    raise ValueError('Incorrect source version or length')
                assert os.pwrite(descriptor, response.content, start) == len(response.content)
                return start
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)

    started = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=256) as pool:
            futures = [pool.submit(fetch, start) for start in missing]
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                completed.add(future.result())
                state['completed'] = sorted(completed)
                state['chunk_size'] = chunk_size
                temporary = state_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(state) + '\n')
                temporary.replace(state_path)
                if count % 8 == 0:
                    print('Missing inference ranges: %d/%d; %.2f MiB/s' % (
                        count, len(missing), count * chunk_size / (time.monotonic() - started) / 1024 ** 2), flush=True)
    finally:
        os.close(descriptor)
    new_header = {'__metadata__': {'format': 'pt', 'source_etag': ETAG,
                                  'source': 'MegaScenes ZeroNVS iter_90000; non-EMA tensors only'}}
    position = 0
    for name, value in selected.items():
        length = value['data_offsets'][1] - value['data_offsets'][0]
        new_header[name] = {'dtype': value['dtype'], 'shape': value['shape'],
                            'data_offsets': [position, position + length]}
        position += length
    serialized = json.dumps(new_header, separators=(',', ':')).encode()
    serialized += b' ' * (-len(serialized) % 8)
    output = DIRECTORY / 'model-inference.safetensors'
    temporary = output.with_suffix('.part')
    with partial.open('rb') as source, temporary.open('wb') as destination:
        destination.write(struct.pack('<Q', len(serialized)))
        destination.write(serialized)
        for name, value in selected.items():
            start, end = value['data_offsets']
            source.seek(data_start + start)
            remaining = end - start
            while remaining:
                content = source.read(min(remaining, 8 * 1024 * 1024))
                if not content:
                    raise EOFError(name)
                destination.write(content)
                remaining -= len(content)
    temporary.rename(output)
    digest = hashlib.sha256()
    with output.open('rb') as stream:
        for content in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(content)
    (DIRECTORY / 'model-inference.safetensors.sha256').write_text(digest.hexdigest() + '  model-inference.safetensors\n')
    manifest = {'source_url': URL, 'source_size': SIZE, 'source_etag': ETAG,
                'tensor_count': len(selected), 'local_sha256': digest.hexdigest(),
                'verification': 'HTTPS range responses, source ETag and sizes checked; selected tensor bytes copied unchanged. Full original multipart ETag not recomputed because EMA bytes were not downloaded.'}
    (DIRECTORY / 'inference_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Inference checkpoint ready:', output, output.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
