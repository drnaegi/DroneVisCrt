"""Match in-line and off-axis source pairs by camera geometry on the same test target."""
from itertools import combinations

import numpy as np

from relation_nvs.io import camera_dict, write_json
from tools.benchmark_stereo_transport import OUT, ROOT, matrixcity_frames


def prepare():
    train = list(matrixcity_frames('train'))
    groups = {'matrix_collinear': [], 'matrix_transverse': []}
    excluded = []
    for target in matrixcity_frames('test'):
        t = target[1].camera_to_world
        sources = [s for s in train if np.allclose(s[1].camera_to_world[:3,:3], t[:3,:3], atol=.001)]
        inline, transverse = [], []
        for a,b in combinations(sources,2):
            # Canonical orientation lets the same pair direction match across grid passes.
            a,b = sorted((a,b), key=lambda s: tuple(np.round(s[1].camera_to_world[:2,3], 1)))
            p = a[1].camera_to_world[:3,3]
            axis = b[1].camera_to_world[:3,3] - p
            length = np.linalg.norm(axis)
            if not 1 < length <= 220:
                continue
            f = (t[:3,3] - p) @ axis / length**2
            d = np.linalg.norm(t[:3,3] - p - f*axis)
            if .15 <= f <= .85:
                pair = (axis, float(f), float(d), [a,b])
                if d <= .2:
                    inline.append(pair)
                elif 20 <= d <= 85:
                    transverse.append(pair)
        matched = [(a,b) for a in inline for b in transverse
                   if np.linalg.norm(a[0]-b[0]) <= .2 and abs(a[1]-b[1]) <= .05]
        if not matched:
            excluded.append(target[1].camera_id)
            continue
        # Choose the shortest common baseline, then the smallest time and distance offsets.
        a,b = min(matched, key=lambda p: (round(np.linalg.norm(p[0][0]),2),
                                          abs(p[0][1]-p[1][1]),p[1][2]))
        for group, (axis, f, offset, pair) in zip(groups, (a,b)):
            rows = pair+[target]
            groups[group].append(dict(paths=[str(p.relative_to(ROOT)) for p,c in rows],
                                      cameras=[camera_dict(c) for p,c in rows],
                                      baseline_m=float(np.linalg.norm(axis)), fraction=f, offset_m=offset))
    for group, rows in groups.items():
        write_json(OUT/(group+'_cases.json'), rows)
    write_json(OUT/'matrix_pose_eligibility.json',dict(matched_targets=len(groups['matrix_collinear']),excluded=excluded))


if __name__ == '__main__':
    prepare()
