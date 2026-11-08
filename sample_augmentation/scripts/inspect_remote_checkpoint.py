"""Inspect the official checkpoint ZIP through validated HTTP byte ranges."""
import io
import json
import pathlib
import pickle
import collections
import urllib.request
import zipfile

URL = 'https://drive.usercontent.google.com/download?id=17WEMfs2HABJcdf4JmuIM3ti0uz37lSZg&export=download&confirm=t'


class RemoteFile(io.RawIOBase):
    def __init__(self, url=URL):
        self.url, self.pos = url, 0
        self.cache = {}
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-63'}), timeout=60) as r:
            if r.status != 206:
                raise RuntimeError('Server must support byte ranges')
            self.size = int(r.headers['Content-Range'].split('/')[-1])
        self.bytes_read = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b''
        key = (self.pos, n)
        if key not in self.cache:
            print('Range read:', self.pos, n, flush=True)
            if n > 8 * 1024 * 1024:
                raise RuntimeError('Metadata inspection requested a large tensor payload')
            end = self.pos + n - 1
            request = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
            with urllib.request.urlopen(request, timeout=90) as r:
                expected = f'bytes {self.pos}-{end}/{self.size}'
                if r.status != 206 or r.headers.get('Content-Range') != expected:
                    raise RuntimeError(f'Unexpected range: {r.status} {r.headers.get("Content-Range")} != {expected}')
                data = r.read()
                if len(data) != n:
                    raise RuntimeError('Incomplete range')
                self.bytes_read += len(data)
                if n < 4 * 1024 * 1024:
                    self.cache[key] = data
                else:
                    self.pos += n
                    return data
        self.pos += n
        return self.cache[key]

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


class TensorRef:
    def __init__(self, storage, storage_offset, shape, stride, *args):
        self.storage = storage
        self.storage_offset = storage_offset
        self.shape = shape
        self.stride = stride


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            return TensorRef
        if module == 'torch' and name.endswith('Storage'):
            return name
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module == 'builtins' and name in ('set', 'slice', 'complex'):
            return getattr(__import__(module), name)
        if module == 'pytorch_lightning.callbacks.model_checkpoint' and name == 'ModelCheckpoint':
            return type('ModelCheckpointMetadata', (), {})
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == 'scalar':
            import numpy
            return numpy.core.multiarray.scalar
        if module == 'numpy' and name == 'dtype':
            import numpy
            return numpy.dtype
        if module == '_codecs' and name == 'encode':
            import codecs
            return codecs.encode
        raise pickle.UnpicklingError(f'Unsupported metadata class: {module}.{name}')

    def persistent_load(self, storage_id):
        return storage_id


if __name__ == '__main__':
    folder = pathlib.Path(__file__).resolve().parents[1] / 'weights'
    metadata = folder / 'official_metadata.pkl'
    entries = folder / 'official_zip_index.json'
    remote = None
    if not metadata.exists():
        remote = RemoteFile()
        print('Checkpoint bytes:', remote.size, flush=True)
        archive = zipfile.ZipFile(remote)
        metadata.write_bytes(archive.read('archive/data.pkl'))
        entries.write_text(json.dumps({x.filename: {'header_offset': x.header_offset, 'file_size': x.file_size, 'CRC': x.CRC}
                                      for x in archive.infolist()}), encoding='utf-8')
    zip_index = json.loads(entries.read_text(encoding='utf-8'))
    checkpoint = MetadataUnpickler(io.BytesIO(metadata.read_bytes())).load()
    print('Top-level keys:', list(checkpoint), flush=True)
    state = checkpoint['state_dict']
    records = {}
    for key, value in state.items():
        storage = value.storage
        name = 'archive/data/' + storage[2]
        item = zip_index[name]
        records[key] = {'shape': list(value.shape), 'storage_type': storage[1],
                        'header_offset': item['header_offset'], 'archive_path': name,
                        'nbytes': item['file_size'], 'crc32': item['CRC'],
                        'stride': list(value.stride), 'storage_offset': value.storage_offset}
    out = pathlib.Path(__file__).resolve().parents[1] / 'weights' / 'official_checkpoint_index.json'
    out.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('State tensors:', len(state), 'metadata downloaded:', remote.bytes_read if remote else 0, flush=True)
    for prefix in sorted(set(k.split('.')[0] for k in state)):
        group = {k: v for k, v in records.items() if k.split('.')[0] == prefix}
        print(prefix, len(group), sum(v['nbytes'] for v in group.values()), flush=True)
