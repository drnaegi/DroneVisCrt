"""Recover the five prospectively selected FloodNet target views and source cameras."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import zipfile
import zlib

from floodnet_joint_data import camera, signed_url
from relation_nvs.archive import RangeArchive, safe_member
from relation_nvs.io import camera_dict, write_json, sha256
from relation_nvs.aerial_mosaic import nadir_camera
from tools.benchmark_stereo_transport import OUT, ROOT, read

TRIPLETS = [(6332,6334,6333),(6433,6435,6434),(6520,6522,6521),
            (6949,6951,6950),(9110,9114,9112)]


def prepare():
    weights = OUT/'band_weights/deployment.json'
    if sha256(weights) != '1e856f6ee5a7d52d7ed7b8a305ceaf5037db61f805332d33ee1c46101e0a71cd':
        raise ValueError('The registered four-region weights have changed')
    data = ROOT/'datasets/floodnet_joint'
    ids = sorted({i for row in TRIPLETS for i in row})
    metadata = {i:read(data/'headers'/f'{i}.json') for i in ids}
    def path(i):
        return data/'images'/f'{i}.jpg'
    missing = [i for i in ids if not path(i).is_file() or zlib.crc32(path(i).read_bytes())!=metadata[i]['zip_crc32']]
    if missing:
        url = signed_url()
        def recover(i):
            with RangeArchive(url) as ar,zipfile.ZipFile(ar) as archive:
                blob = safe_member(archive,metadata[i]['member'])
            if zlib.crc32(blob)!=metadata[i]['zip_crc32']:
                raise ValueError(f'Image CRC mismatch for {i}')
            with tempfile.NamedTemporaryFile(dir=path(i).parent,delete=False) as handle:
                handle.write(blob)
            Path(handle.name).replace(path(i))
        with ThreadPoolExecutor(6) as pool:
            list(pool.map(recover,missing))
    rows = []
    for ids in TRIPLETS:
        cams = [nadir_camera(camera(metadata[i],metadata[ids[0]]),metadata[i]['xmp']['GimbalRollDegree']) for i in ids]
        rows.append(dict(paths=[str(path(i).relative_to(ROOT)) for i in ids],cameras=[camera_dict(c) for c in cams]))
    write_json(OUT/'flood_holdout_cases.json',rows)


if __name__ == '__main__':
    prepare()
