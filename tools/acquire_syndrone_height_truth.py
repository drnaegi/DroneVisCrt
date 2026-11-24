"""Read height-transfer geometry and semantic labels only for post-render evaluation."""
from concurrent.futures import ThreadPoolExecutor
import zipfile

from relation_nvs.archive import RangeArchive,safe_member
from relation_nvs.io import sha256,write_json
from tools.acquire_syndrone_height_transfer import TOWNS,FRAMES,HEIGHTS,DATA


def acquire(town):
    folder=DATA/town/'evaluation'
    folder.mkdir(exist_ok=True)
    hashes={}
    for kind,modality,heights in [('color','semantic',(50,)),('depth','depth',HEIGHTS)]:
        url=f'https://lttm.dei.unipd.it/paper_data/syndrone/{town}_Opt_120_{kind}.zip'
        with RangeArchive(url) as remote,zipfile.ZipFile(remote) as archive:
            for f in FRAMES:
                for h in heights:
                    path=folder/f'{f}_{h}_{modality}.png'
                    member=f'{town}_Opt_120/ClearNoon/height{h}m/{modality}/{f}.png'
                    path.write_bytes(safe_member(archive,member))
                    hashes[path.name] = dict(member=member,url=url,sha256=sha256(path))
    write_json(folder/'members.json',hashes)
    print(town,flush=True)


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(acquire,TOWNS))
