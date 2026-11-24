"""Render new downward-facing views from a JSON manifest with two calibrated RGB references."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from relation_nvs.aerial_mosaic import AerialMosaic, SurveyCalibration, nadir_camera
from relation_nvs.io import load_camera, sha256, write_json

POINTMAP_METHODS = tuple('surface_pointmap_'+rule for rule in ('median','scale','rays','matches','similarity','refined','source_fit','rays_rescaled'))
MONOCULAR_SURFACES = ('surface_detail','surface_detail_median','surface_precision','surface_precision_depth','surface_learned')
SURFACE_METHODS = MONOCULAR_SURFACES + POINTMAP_METHODS


def run(manifest, output, method='plane', band_weights=None):
    if method in POINTMAP_METHODS:
        from relation_nvs.survey_pointmap import pointmap_precision
        with pointmap_precision():
            return _run(manifest, output, method, band_weights)
    return _run(manifest, output, method, band_weights)


def _run(manifest, output, method, band_weights):
    spec = json.loads(manifest.read_text())
    references = spec['references']
    if len(references) != 2:
        raise ValueError('Exactly two source observations required')
    cameras = [load_camera(item['camera']) for item in references]
    targets = [load_camera(item) for item in spec['targets']]
    if not targets or len({c.camera_id for c in targets}) != len(targets):
        raise ValueError('Provide targets with distinct camera IDs')
    images, hashes = [], []
    for item, camera in zip(references, cameras):
        path = manifest.parent / item['rgb']
        with Image.open(path) as im:
            if im.width * camera.height != im.height * camera.width:
                raise ValueError('Source and calibration aspect ratios differ')
            images.append(np.asarray(im.convert('RGB').resize((camera.width, camera.height), Image.Resampling.LANCZOS)))
        hashes.append(sha256(path))
    rolls = spec.get('dji_gimbal_roll_deg', {})
    cameras = [nadir_camera(c, rolls.get(c.camera_id)) for c in cameras]
    pose_source = spec.get('pose_source', 'survey_telemetry')
    if pose_source == 'survey_telemetry':
        calibration = SurveyCalibration.estimate(images, cameras)
    elif pose_source == 'calibrated':
        calibration = SurveyCalibration(cameras[0].camera_to_world[:2,3], np.eye(2), 0, None)
    else:
        raise ValueError('pose_source must be survey_telemetry or calibrated')
    cameras = [calibration.apply(c) for c in cameras]
    targets = [calibration.apply(nadir_camera(c, rolls.get(c.camera_id))) for c in targets]
    if method == 'surface_learned':
        if band_weights is None:
            band_weights = Path(__file__).resolve().parents[1]/'outputs/aerial_stereo_study/band_weights/deployment.json'
        coefficients = json.loads(band_weights.read_text())['weights']['geometry']
    if method == 'plane':
        renderer = AerialMosaic.from_views(images, cameras)
    else:
        from relation_nvs.survey_flow import estimate_flow, get_flow_model
        from relation_nvs.stereo_transport import SurveyStereo, rectify_source_pair, unrectify_flow
        if images[0].shape != images[1].shape:
            raise ValueError('Source rasters must have equal dimensions for dense flow')
        flow_model = get_flow_model()
        if method in POINTMAP_METHODS:
            from relation_nvs.survey_pointmap import load_pair_model, ordered_pointmaps, pointmap_renderer
            pair_model = load_pair_model()
        if method == 'rectified':
            partners, homographies = rectify_source_pair(images, cameras)
            flow = unrectify_flow(estimate_flow(flow_model, images, partners), cameras, homographies)
        elif method == 'surface_pointmap_refined':
            flow = None
        elif method in ('transport','depth_prior',*SURFACE_METHODS):
            flow = estimate_flow(flow_model, images)
        else:
            raise ValueError(method)
        if method in POINTMAP_METHODS:
            pair = ordered_pointmaps(pair_model, images)
            if method == 'surface_pointmap_refined':
                from relation_nvs.survey_pointmap import calibrated_pair_depth, rematch_surface
                depths, alignment = calibrated_pair_depth(pair,cameras,None,'rays')
                surface_flow,_ = rematch_surface(images,cameras,depths,flow_model)
                renderer = SurveyStereo(images,cameras,surface_flow,depths,'metric_pair')
            else:
                renderer, alignment = pointmap_renderer(images,cameras,flow,pair,method.removeprefix('surface_pointmap_'))
        else:
            depths = None
            if method in ('depth_prior',*MONOCULAR_SURFACES):
                from relation_nvs.survey_depth import infer_depth, load_moge
                depth_model = load_moge()
                depths = [infer_depth(depth_model, image, c) for image,c in zip(images,cameras)]
            rule = {'surface_detail_median':'median_stereo','surface_precision':'precision',
                    'surface_precision_depth':'precision'}.get(method,'qualified')
            renderer = SurveyStereo(images,cameras,flow,monocular=depths,prior_rule=rule)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent) as tmp:
        dest = Path(tmp) / 'result'
        dest.mkdir()
        results = []
        for target in targets:
            if method == 'surface_precision':
                image, provenance = renderer.render(target,precision_donor=True)
            else:
                image, provenance = renderer.render(target)[:2]
            if method in SURFACE_METHODS:
                from relation_nvs.survey_visibility import render_surfaces, combine_surface_and_inverse
                _, blended, _ = render_surfaces(renderer,target)
                if method == 'surface_learned':
                    from relation_nvs.survey_bands import combine
                    image = combine(blended,image,[*cameras,target],coefficients)
                else:
                    image, _ = combine_surface_and_inverse(blended,image)
            folder = dest / target.camera_id
            folder.mkdir()
            Image.fromarray(image).save(folder / 'rgb.png')
            Image.fromarray(provenance).save(folder / 'source.png')
            results.append(dict(camera_id=target.camera_id, coverage=float((provenance > 0).mean()),
                                rgb_sha256=sha256(folder / 'rgb.png')))
        source_map_role = 'RGB donor'
        if method in SURFACE_METHODS:
            source_map_role = 'inverse detail branch only'
        if method == 'surface_learned':
            source_map_role = 'inverse branch only; each frequency band may mix branches'
        write_json(dest / 'manifest.json', dict(input_sha256=sha256(manifest), source_rgb_sha256=hashes,
                   horizontal_similarity=calibration.correction.tolist(), matches=calibration.inliers, scatter_m=calibration.scatter_m,
                   provenance={'0': 'reflected context', '1': 'first reference', '2': 'second reference'}, views=results, method=method, pose_source=pose_source, source_map_role=source_map_role, depth_models=getattr(renderer,'prior_fits',[]),
                   **({'pointmap_alignment':alignment} if method in POINTMAP_METHODS else {}),
                   **({'band_weights_sha256':sha256(band_weights)} if method=='surface_learned' else {})))
        dest.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--method', choices=['plane','transport','rectified','depth_prior',*SURFACE_METHODS], default='plane')
    parser.add_argument('--band-weights',type=Path,help='Region-held-out or deployment coefficients for surface_learned')
    args = parser.parse_args()
    run(args.manifest, args.output, args.method, args.band_weights)
