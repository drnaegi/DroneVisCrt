"""Cache the supervised archive camera headers and class masks without using target classes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import struct
import zipfile
import zlib

from acquire_floodnet_supervised_pairs import OUT as DEV, RangeArchive, signed_url, header, write_json
from relation_nvs.io import sha256

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'datasets/floodnet_joint'
OUT = Path(os.environ.get('FLOODNET_JOINT_OUTPUT', ROOT / 'outputs/floodnet_joint_pairs')).resolve()
def main():
    cache = CACHE
    for name in ('headers','masks','images'):
        (cache/name).mkdir(parents=True,exist_ok=True)
    url = signed_url()
    with RangeArchive(url) as ar, zipfile.ZipFile(ar) as archive:
        entries = [[z.filename,z.header_offset,z.compress_size,z.file_size,z.CRC,z.compress_type,z.flag_bits]
                   for z in archive.infolist() if '-org-img/' in z.filename or '-label-img/' in z.filename]
        write_json(cache/'archive.json',dict(size=ar.size,etag=ar.etag,entries=entries))
        masks = [e for e in entries if e[0].startswith('train/train-label-img/')]
        start = min(e[1] for e in masks)
        end = max(e[1]+e[2]+512 for e in masks)
        ar.seek(start)
        block = ar.read(end-start)
        for name,offset,size,length,crc,method,_ in masks:
            pos = offset-start
            ns,extra = struct.unpack_from('<HH',block,pos+26)
            if block[pos+30:pos+30+ns].decode()!=name:
                raise ValueError(name)
            pos += 30+ns+extra
            data=zlib.decompress(block[pos:pos+size],-15)
            if zlib.crc32(data)!=crc or len(data)!=length:
                raise ValueError(name)
            (cache/'masks'/Path(name).name).write_bytes(data)
    import json
    legacy=DEV.parent/'all_labeled_cameras'
    images=[e for e in entries if '-org-img/' in e[0]]
    def collect(group):
        with RangeArchive(url) as ar:
            for e in group:
                path=cache/'headers'/Path(e[0]).with_suffix('.json').name
                if path.exists():
                    continue
                for directory in (DEV/'headers',legacy):
                    prior=directory/path.name
                    if prior.exists():
                        record=json.loads(prior.read_text())
                        if record['zip_crc32']==e[4]:
                            record['member']=e[0]
                            write_json(path,record)
                            break
                else:
                    for length in (65536,262144,1048576):
                        ar.seek(e[1])
                        try:
                            record=header(ar.read(min(e[2]+512,length)),e)
                            break
                        except OSError:
                            if length==1048576:
                                raise
                    write_json(path,record)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(collect,[images[i::16] for i in range(16)]))
    print('headers',len(images),'train masks',len(masks))
if __name__=='__main__':
    main()
