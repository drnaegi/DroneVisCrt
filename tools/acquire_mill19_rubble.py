"""Acquire selected CRC-checked members of the pinned Mill 19 COLMAP release."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import struct
import time
from tempfile import TemporaryFile
import zipfile
import zlib

import requests

from tools.benchmark_stereo_transport import OUT, ROOT, read

REVISION = '1115bcd77171e6200f2dbf227c8a0cca3a09deff'
URL = f'https://hf-mirror.com/datasets/HexuZhao/mega_nerf_rubble_colmap/resolve/{REVISION}/rubble-colmap.zip'
SIZE = 16267992576
DATA = ROOT/'datasets/stereo_survey/mill19_rubble'
PREFIX = 'rubble-colmap/'
CHUNK = 4*1024**2


def fetch(start, stop):
    for attempt in range(5):
        try:
            interval = f'{start}-{stop-1}'
            with requests.get(URL,params={'offset':start,'stop':stop},headers={'Range':'bytes='+interval},
                              timeout=(15,60)) as reply:
                reply.raise_for_status()
                if reply.headers.get('Content-Range')!=f'bytes {interval}/{SIZE}' or len(reply.content)!=stop-start:
                    raise ValueError('Incomplete archive range')
                return reply.content
        except (requests.RequestException,ValueError):
            if attempt==4:
                raise
            time.sleep(2**attempt)


def directory():
    # ZIP64 stores offsets against the original file. Sparse prefix contains no member bytes.
    length=3*1024**2
    with TemporaryFile() as temp:
        temp.seek(SIZE-length)
        temp.write(fetch(SIZE-length,SIZE))
        with zipfile.ZipFile(temp) as archive:
            return {i.filename:i for i in archive.infolist()}


def member(name, entry, pool):
    path=DATA/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file() and path.stat().st_size==entry.file_size:
        with path.open('rb') as f:
            crc=0
            while block:=f.read(CHUNK):
                crc=zlib.crc32(block,crc)
        if crc==entry.CRC:
            return
    offset=entry.header_offset
    header=fetch(offset,offset+30)
    if header[:4]!=b'PK\x03\x04' or entry.compress_type!=zipfile.ZIP_DEFLATED:
        raise ValueError('Unexpected ZIP header')
    start=offset+30+sum(struct.unpack_from('<HH',header,26))
    stop=start+entry.compress_size
    locations=range(start,stop,CHUNK)
    inflater=zlib.decompressobj(-15)
    crc=0
    temp=path.with_suffix(path.suffix+'.part')
    with temp.open('wb') as f:
        for block in pool.map(lambda loc:fetch(loc,min(loc+CHUNK,stop)),locations):
            raw=inflater.decompress(block)
            f.write(raw)
            crc=zlib.crc32(raw,crc)
    if not inflater.eof or temp.stat().st_size!=entry.file_size or crc!=entry.CRC:
        raise ValueError(f'CRC or size mismatch for {name}')
    temp.replace(path)
    print(name,entry.file_size,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['poses','images'])
    parser.add_argument('--repeat',action='store_true')
    args=parser.parse_args()
    stage=args.stage
    if stage=='poses':
        names=['sparse/0/cameras.bin','sparse/0/images.bin']
    else:
        selected=read(OUT/('mill19_rubble_repeat_selection.json' if args.repeat else 'mill19_rubble_selection.json'))['selected']
        names=sorted({'images/'+name for row in selected for name in row['names']})
    index=directory()
    with ThreadPoolExecutor(16) as chunks, ThreadPoolExecutor(6) as members:
        list(members.map(lambda n:member(n,index[PREFIX+n],chunks),names))
