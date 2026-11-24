"""Compare calibrated and fixed detail transfer using previously sealed source-only renders."""
import argparse

import numpy as np
from PIL import Image
from relation_nvs.survey_visibility import combine_surface_and_inverse
from relation_nvs.io import sha256,write_json
from tools.benchmark_stereo_transport import read,OUT


def run(group):
    directory=OUT/group
    records={}
    for key,inv in [('qualified','depth_prior'),('median_stereo','median_stereo')]:
        mesh='mesh_'+key+'_blend'
        a,b=read(directory/(inv+'.json')),read(directory/(mesh+'.json'))
        if a.keys()!=b.keys():raise ValueError('Incomplete pair of source-only render sets')
        for case in a:
            images=[]
            for name,ref in [(inv,a),(mesh,b)]:
                path=directory/case/(name+'.png')
                if sha256(path)!=ref[case]['rgb_sha256']:raise ValueError(path)
                with Image.open(path) as im:images.append(np.asarray(im.convert('RGB')))
            fixed,adaptive=combine_surface_and_inverse(images[1],images[0])
            for name,image in [('fixed',fixed),('adaptive',adaptive)]:
                method='detail_'+key+'_'+name
                output=directory/case/(method+'.png')
                Image.fromarray(image).save(output)
                records.setdefault(method,{})[case]=dict(rgb_sha256=sha256(output))
    for name,records_by_case in records.items():
        write_json(directory/(name+'.json'),records_by_case)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups',nargs='+')
    for group in parser.parse_args().groups:run(group)
