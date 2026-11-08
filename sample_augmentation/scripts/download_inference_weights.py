"""Fetch only inference tensors from the unmodified official ZIP checkpoint.

Every selected tensor is checked against the ZIP CRC before conversion. UNet
weights use the same EMA replacement as the official guidance loader.
"""
import concurrent.futures
import hashlib
import json
import pathlib
import struct
import time
import urllib.request
import zlib

from inspect_remote_checkpoint import URL

ROOT = pathlib.Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'weights'
CACHE = FOLDER / 'range_cache'
CHUNK = 32 * 1024 * 1024


def main():
    import torch
    from safetensors.torch import save_file
    index = json.loads((FOLDER / 'official_checkpoint_index.json').read_text())
    selected = {}
    for name, record in index.items():
        source_name = name
        if name.startswith('model_ema.'):
            continue
        if name.startswith('model.'):
            source_name = 'model_ema.' + name[len('model.'):].replace('.', '')
        selected[name] = dict(index[source_name], source_name=source_name)
    ranges = []
    for item in sorted(selected.values(), key=lambda x: x['header_offset']):
        start = item['header_offset']
        end = start + 256 + item['nbytes']
        if ranges and start - ranges[-1][1] <= 65536:
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])
    jobs = []
    for start, end in ranges:
        jobs.extend((a, min(a + CHUNK, end)) for a in range(start, end, CHUNK))
    CACHE.mkdir(exist_ok=True)
    partials = list(FOLDER.glob('zeronvs.ckpt*.part'))
    partial = max(partials, key=lambda p: p.stat().st_size) if partials else None
    partial_size = partial.stat().st_size if partial else 0
    total = sum(b - a for a, b in jobs)
    print(f'Inference tensors: {len(selected)}; transfer {total / 2**30:.2f} GiB in {len(jobs)} ranges', flush=True)

    def fetch(job):
        a, b = job
        path = CACHE / f'{a}_{b}.bin'
        if path.exists() and path.stat().st_size == b - a:
            return b - a
        if partial and b <= partial_size:
            with partial.open('rb') as f:
                f.seek(a)
                data = f.read(b - a)
        else:
            for attempt in range(5):
                try:
                    request = urllib.request.Request(URL, headers={'Range': f'bytes={a}-{b-1}'})
                    with urllib.request.urlopen(request, timeout=120) as r:
                        if r.status != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {a}-{b-1}/'):
                            raise RuntimeError('Incorrect HTTP byte range')
                        data = r.read()
                    if len(data) != b - a:
                        raise RuntimeError('Incomplete HTTP byte range')
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 16))
        temp = path.with_suffix('.tmp')
        temp.write_bytes(data)
        temp.replace(path)
        return len(data)

    completed = 0
    start_time = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i, n in enumerate(pool.map(fetch, jobs), 1):
            completed += n
            if i % 8 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} ranges; {completed/2**30:.2f}/{total/2**30:.2f} GiB; {time.monotonic()-start_time:.0f}s', flush=True)

    def read_range(start, length):
        end = start + length
        chunks = []
        for a, b in jobs:
            left, right = max(a, start), min(b, end)
            if left < right:
                with (CACHE / f'{a}_{b}.bin').open('rb') as f:
                    f.seek(left - a)
                    chunks.append(f.read(right - left))
        data = b''.join(chunks)
        if len(data) != length:
            raise RuntimeError('Missing cached data')
        return data

    dtypes = {'FloatStorage': torch.float32, 'LongStorage': torch.int64,
              'HalfStorage': torch.float16, 'DoubleStorage': torch.float64,
              'IntStorage': torch.int32}
    state = {}
    for name, item in selected.items():
        header = read_range(item['header_offset'], 30)
        if header[:4] != b'PK\x03\x04':
            raise RuntimeError('Bad ZIP local header')
        name_len, extra_len = struct.unpack_from('<HH', header, 26)
        offset = item['header_offset'] + 30 + name_len + extra_len
        data = bytearray(read_range(offset, item['nbytes']))
        if zlib.crc32(data) & 0xffffffff != item['crc32']:
            raise RuntimeError(f'Tensor CRC mismatch: {name}')
        tensor = torch.frombuffer(data, dtype=dtypes[item['storage_type']])
        state[name] = torch.as_strided(tensor, item['shape'], item['stride'], item['storage_offset']).clone()
    destination = FOLDER / 'zeronvs_ema_inference.safetensors'
    save_file(state, str(destination), metadata={'source': URL, 'unet': 'official EMA', 'precision': 'original FP32'})
    digest = hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest()
    manifest = {'source_url': URL, 'source_checkpoint_bytes': 15466104283,
                'extraction': 'Original FP32 tensors; every tensor ZIP CRC32 checked; EMA UNet replaces raw UNet, matching official loader',
                'tensor_count': len(state), 'sha256': digest, 'path': destination.name,
                'tensor_sources': {k: v['source_name'] for k, v in selected.items()}}
    (FOLDER / 'inference_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Verified and saved:', destination, 'SHA256:', digest, flush=True)


if __name__ == '__main__':
    main()
