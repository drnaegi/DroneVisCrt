"""Paired pointmap results with fixed targets, common comparator support and equal-region averaging."""
import argparse
import numpy as np

from relation_nvs.io import sha256, write_json
from tools.benchmark_stereo_transport import OUT, ROOT, PAIR_METHODS, read
from tools.summarize_survey_nvs import GEOGRAPHY, METRICS


def summarize(groups):
    rows, geo, digests = [], {}, {}
    grouped_outputs = {}
    for group in groups:
        folder = OUT/group
        paths = [folder/'scores.json',folder/'pointmap_scores.json']
        paths += [folder/f for f in ('source_fit_scores.json','optimizer_1_scores.json','optimizer_10_scores.json','optimizer_30_scores.json','viewcrafter_scores.json','global_optimizer_scores.json','rematch_scores.json','similarity_scores.json','mapanything_scores.json','mapanything_scale_scores.json') if (folder/f).is_file()]
        manifest = read(folder/'predictions.json')
        truth = {r['case']:sha256(ROOT/r['paths'][2]) for r in manifest}
        by_method = {}
        digests[group] = {p.name:sha256(p) for p in paths}
        for path in paths:
            for r in read(path)['rows']:
                case, method = r['case'], r['method']
                if r['target_sha256'] != truth[case]:
                    raise ValueError('Compared targets differ')
                values = np.array([r[m] for m in METRICS])
                values_before = by_method.setdefault(method,{}).setdefault(case,values)
                if not np.allclose(values_before,values,atol=1e-7,rtol=0):
                    raise ValueError('Repeated scores differ')
        grouped_outputs[group] = {m:dict(zip(METRICS,np.mean(list(v.values()),axis=0))) for m,v in by_method.items()}
        for candidate in (*PAIR_METHODS,*[m for m in ('pointmap_source_fit','pointmap_rays_rescaled','surface_refined','surface_selected','pointmap_similarity','mapanything_rays') if m in by_method]):
            if set(by_method[candidate]) != set(truth):
                raise ValueError('Incomplete candidate')
            for ctrl in (*PAIR_METHODS,'ground_selected','pointmap_rays_rescaled','pointmap_optimizer_1','pointmap_optimizer_10','pointmap_optimizer_30','viewcrafter_sparse_rgb','mapanything_metric','mapanything_median','mapanything_rays','pointmap_similarity','moge_matched_precision','detail_median_stereo_fixed','pointmap_global_optimizer','rife_ground','depthsplat_dl3dv_256x448_plane_bg'):
                if ctrl==candidate or ctrl not in by_method:
                    continue
                cases = sorted(by_method[ctrl])
                if len(cases)!=len(truth):
                    if ctrl!='rife_ground' or set(read(folder/'rife_ground_eligibility.json')['eligible'])!=set(cases):
                        raise ValueError('Unexplained missing case')
                difference = np.stack([by_method[candidate][c]-by_method[ctrl][c] for c in cases])
                delta = difference.mean(0)
                rows.append(dict(group=group,candidate=candidate,control=ctrl,n=len(cases),
                                 delta=dict(zip(METRICS,delta)),wins=dict(zip(METRICS,((difference*[1,1,-1])>0).sum(0).tolist()))))
                if group in GEOGRAPHY:
                    geo.setdefault((candidate,ctrl),{}).setdefault(GEOGRAPHY[group],[]).append(delta)
    aggregate = []
    for (candidate,ctrl),by_region in geo.items():
        sites = {region:dict(zip(METRICS,np.mean(v,axis=0))) for region,v in by_region.items()}
        aggregate.append(dict(candidate=candidate,control=ctrl,sites=sites,delta={m:np.mean([s[m] for s in sites.values()]) for m in METRICS}))
    write_json(OUT/'pointmap_summary.json',dict(score_sha256=digests,means=grouped_outputs,per_group=rows,equal_geography=aggregate))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('groups',nargs='+')
    summarize(p.parse_args().groups)
