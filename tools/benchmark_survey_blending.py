"""Matched controls for the two frequency bands, using identical frozen renderings."""
import argparse

import cv2
import numpy as np
from PIL import Image

from relation_nvs.io import sha256, write_json
from tools.benchmark_stereo_transport import OUT, read


def run(group):
    directory = OUT / group
    mesh = read(directory / 'mesh_qualified_blend.json')
    inverse = read(directory / 'depth_prior.json')
    if mesh.keys() != inverse.keys():
        raise ValueError('Surface and inverse renders cover different targets')
    records = {}
    for case in mesh:
        images = []
        for name, manifest in [('mesh_qualified_blend', mesh), ('depth_prior', inverse)]:
            path = directory / case / (name + '.png')
            if sha256(path) != manifest[case]['rgb_sha256']:
                raise ValueError(f'Changed source render {path}')
            with Image.open(path) as im:
                images.append(np.asarray(im.convert('RGB'), dtype=np.float32))
        surface, single = images
        low_surface, low_single = [cv2.GaussianBlur(im, (0, 0), 1.5) for im in images]
        controls = {
            'simple_average': (surface + single) / 2,
            'reverse_bands': low_single + (surface - low_surface),
            'sharpen_surface': surface + (surface - low_surface),
        }
        for name, rgb in controls.items():
            path = directory / case / (name + '.png')
            Image.fromarray(rgb.clip(0, 255).round().astype(np.uint8)).save(path)
            records.setdefault(name, {})[case] = {'rgb_sha256': sha256(path)}
    for name, rows in records.items():
        write_json(directory / (name + '.json'), rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups', nargs='+')
    for group in parser.parse_args().groups:
        run(group)
