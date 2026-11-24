"""Acquire three predetermined height triplets per town for out-of-town NVS evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import zipfile

from PIL import Image
from relation_nvs.archive import RangeArchive, safe_member
from relation_nvs.io import camera_dict, write_json, sha256
from relation_nvs.syndrone import carla_camera
from tools.benchmark_stereo_transport import ROOT, OUT, read

TOWNS = ('Town02','Town05','Town06','Town10HD')
FRAMES = ('00499','01499','02499')
HEIGHTS = (20,80,50)
DATA = ROOT/'datasets/stereo_survey/syndrone_height_transfer'


def acquire(town, frames=FRAMES, data=DATA):
    url = f'https://lttm.dei.unipd.it/paper_data/syndrone/{town}_Opt_120_color.zip'
    folder = data/town
    folder.mkdir(parents=True,exist_ok=True)
    with RangeArchive(url) as remote, zipfile.ZipFile(remote) as archive:
        rows = []
        for frame in frames:
            paths, cameras, hashes = [], [], {}
            for h in HEIGHTS:
                stem = f'{town}_Opt_120/ClearNoon/height{h}m'
                for kind, ext in [('rgb','jpg'),('camera','json')]:
                    member = f'{stem}/{kind}/{frame}.{ext}'
                    path = folder / f'{town}_{frame}_{h}.{ext}'
                    data = safe_member(archive,member)
                    path.write_bytes(data)
                    hashes[member] = sha256(path)
                    if kind=='rgb':
                        paths.append(str(path.relative_to(ROOT)))
                        with Image.open(path) as im:
                            width,height = im.size
                    else:
                        cam = carla_camera(read(path),f'{town}_{frame}_{h}',width,height).resized(512,288)
                        cameras.append(camera_dict(cam))
            rows.append(dict(paths=paths,cameras=cameras,hashes=hashes,town=town,frame=frame,source_archive=url))
            print(town,frame,flush=True)
        write_json(folder/'members.json',dict(rows=rows,etag=remote.etag))
        return rows


def main(repeat=False):
    frames=('00749','01749','02749') if repeat else FRAMES
    group='syndrone_repeat' if repeat else 'syndrone_transfer'
    data=DATA.parent/'syndrone_height_repeat' if repeat else DATA
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [r for town_rows in pool.map(lambda town:acquire(town,frames,data),TOWNS) for r in town_rows]
    write_json(OUT/(group+'_cases.json'),rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat',action='store_true')
    main(parser.parse_args().repeat)
