"""Compare the archived RGB-only diffusion middle frame at the survey evaluation size."""
import numpy as np
from PIL import Image

from relation_nvs.io import load_camera, sha256, write_json
from tools.benchmark_stereo_transport import OUT, ROOT, read, score

METHOD='viewcrafter_sparse_rgb'


def main():
    upstream=ROOT/'outputs/relation_nvs_20261120'/METHOD
    group='syndrone_heights'
    dest=OUT/group
    spec=read(upstream/'protocol_before_generation.json')
    records={}
    for row in read(dest/'predictions.json'):
        town=row['case'].split('_')[0]
        case=next(c for c in spec['cases'] if c['town']==town)
        folder=upstream/town
        status=read(folder/'generation_status.json')
        small=folder/'viewcrafter_sparse.png'
        mid=folder/'generated'/f"{spec['target_frame']:03}.png"
        if status['status']!='complete' or status['target_truth_read'] or status['source_depth_or_semantics_read']:
            raise ValueError('ViewCrafter inputs differ from the RGB-only protocol')
        if sha256(small)!=status['output_sha256'] or [x['rgb_sha256'] for x in case['references']]!=row['source_hashes']:
            raise ValueError('Changed RGB')
        target=load_camera(row['cameras'][2])
        for expected,actual in zip([*[v['camera'] for v in case['references']],case['cameras'][spec['target_frame']]],row['cameras']):
            camera=load_camera(expected).resized(actual['width'],actual['height'])
            if not (np.allclose(camera.camera_to_world,actual['camera_to_world'],atol=1e-7) and np.allclose(camera.intrinsics,actual['intrinsics'],atol=1e-6)):
                raise ValueError('Mismatched cameras')
        with Image.open(mid) as image, Image.open(small) as reduced:
            if not np.array_equal(image.resize(reduced.size,Image.Resampling.BOX),reduced):
                raise ValueError('Middle frame differs from audited diffusion result')
            output=dest/row['case']/(METHOD+'.png')
            image.convert('RGB').resize((target.width,target.height),Image.Resampling.LANCZOS).save(output)
        records[row['case']]=dict(rgb_sha256=sha256(output),native_frame_sha256=sha256(mid),prior_audit_sha256=sha256(small))
    write_json(dest/(METHOD+'.json'),records)
    score(group,only=(METHOD,),output='viewcrafter_scores.json')


if __name__=='__main__':
    main()
