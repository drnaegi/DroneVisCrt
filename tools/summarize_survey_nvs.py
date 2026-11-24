"""Summarize matched survey NVS predictions and check each target identity across methods."""
import argparse
import csv
from pathlib import Path

import numpy as np

from relation_nvs.io import sha256, write_json
from tools.benchmark_stereo_transport import OUT, read

METRICS = ('psnr','ssim','lpips')
GEOGRAPHY = {'monterey':'Monterey','cross':'Sheffield','park1':'Sheffield','park3':'Sheffield',
             'noema':'Noema','sand_key':'Sand Key'}


def summarize(groups):
    by_group, summary, masked = {}, [], []
    for group in groups:
        folder = OUT/group
        predictions = read(folder/'predictions.json')
        scores = read(folder/'scores.json')
        target_ids = {r['case']:r['paths'][-1] for r in predictions}
        cases = list(target_ids)
        all_rows = {}
        for r in scores['rows']:
            key=(r['case'],r['method'])
            if key in all_rows or r['target']!=Path(target_ids[r['case']]).name:
                raise ValueError(f'Duplicate or mismatched evaluation {key}')
            all_rows[key]=r
        for c in cases:
            digests={r['target_sha256'] for (case,_),r in all_rows.items() if case==c}
            if len(digests)!=1: raise ValueError('Reference targets differ across methods')
        by_group[group]={}
        for name in scores['mean']:
            available = [c for c in cases if (c,name) in all_rows]
            if len(available) != len(cases):
                if name not in ('rife','rife_ground'):
                    raise ValueError(f'Missing predictions for {name}')
                eligible = read(folder / (name + '_eligibility.json'))
                if set(available) != set(eligible['eligible']) or set(cases)-set(available) != set(eligible['excluded']):
                    raise ValueError(f'Unexplained interpolation exclusions for {group}')
            rows=[all_rows[c,name] for c in available]
            values=np.array([[r[m] for m in METRICS] for r in rows])
            by_group[group][name]=dict(zip(available,values))
            average=dict(zip(METRICS,values.mean(0)))
            if not np.allclose(values.mean(0),[scores['mean'][name][m] for m in METRICS]):
                raise ValueError('Means differ')
            summary.append(dict(group=group,method=name,targets=len(rows),**average))
            for region in ['observed_psnr','complement_psnr']:
                sample=[r[region] for r in rows if r[region] is not None]
                masked.append(dict(group=group,method=name,region=region,targets=len(sample),
                                   mean_psnr=np.mean(sample) if sample else None,
                                   average_plane_coverage=np.mean([r['plane_support'] for r in rows])))
    for path, rows in [(OUT/'method_means.csv',summary),(OUT/'coverage_psnr.csv',masked)]:
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    contrasts=[]
    for candidate,control in [('learned_bands_geometry',c) for c in ('learned_bands_image','learned_bands_constant',
                              'simple_average','detail_qualified_fixed','rife_ground','depthsplat_dl3dv_256x448_plane_bg')] + [
                              ('precision_donor','detail_median_stereo_fixed'),('precision_depth','detail_median_stereo_fixed'),
                              ('precision_donor','precision_depth'),('precision_donor','detail_qualified_fixed'),
                              ('precision_donor','rife_ground'),('precision_donor','simple_average'),
                              ('depth_prior','median_stereo'),('detail_qualified_fixed','detail_median_stereo_fixed'),
                              ('detail_qualified_fixed','depth_prior'),('detail_qualified_fixed','mesh_qualified_blend'),
                              ('detail_qualified_fixed','detail_qualified_adaptive'),
                              ('detail_qualified_fixed','simple_average'), ('detail_qualified_fixed','reverse_bands'),
                              ('detail_qualified_fixed','sharpen_surface'),
                              ('detail_qualified_fixed','depthsplat_dl3dv_256x448_plane_bg'),
                              ('detail_qualified_fixed','mvsplat_acid_native_plane_bg'),
                              ('detail_qualified_fixed','rife_ground')]:
        geography={}
        for group,methods in by_group.items():
            if control not in methods or candidate not in methods:continue
            common = [c for c in methods[candidate] if c in methods[control]]
            delta=np.stack([methods[candidate][c]-methods[control][c] for c in common])
            contrasts.append(dict(candidate=candidate,control=control,group=group,
                                  targets=len(delta),deltas=dict(zip(METRICS,delta.mean(0))),
                                  wins=dict(zip(METRICS,((delta*[1,1,-1])>0).sum(0).tolist()))))
            if group in GEOGRAPHY:
                geography.setdefault(GEOGRAPHY[group],[]).append(delta.mean(0))
        if geography:
            site_values=[np.mean(v,axis=0) for v in geography.values()]
            contrasts.append(dict(candidate=candidate,control=control,group='equal_geography',geographies=list(geography),
                                  deltas=dict(zip(METRICS,np.mean(site_values,axis=0))),geo_deltas=dict(zip(geography,[dict(zip(METRICS,v)) for v in site_values]))))
    write_json(OUT/'paired_comparisons.json',dict(score_sha256={g:sha256(OUT/g/'scores.json') for g in groups},
               aggregation='Equal targets within flight, equal flight means within region, equal region means; pilot Brighton/FloodNet/MatrixCity excluded from geography means.',
               contrasts=contrasts))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('groups',nargs='+')
    summarize(p.parse_args().groups)
