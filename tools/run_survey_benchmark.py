"""Run the two-source survey benchmark in the workspace, resuming generated method manifests."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from tools.benchmark_stereo_transport import OUT, ROOT, read

FULL_GROUPS = ('fresh','confirmation','brighton','matrixcity','monterey','cross','park1',
               'park3','noema','sand_key','lego','flood_holdout')
POSE_GROUPS = ('matrix_collinear','matrix_transverse')
EXTERNAL = Path('/autodl-tmp/DroneVisCrt')


def execute(args):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT/'tools'), str(args.packages),
                                        env.get('PYTHONPATH','')])
    env.setdefault('OMP_NUM_THREADS','2')
    env.setdefault('OPENBLAS_NUM_THREADS','1')

    def call(module, *options):
        subprocess.run([sys.executable, '-u', '-m', 'tools.' + module, *map(str, options)],
                       cwd=ROOT, env=env, check=True)

    for group in args.groups:
        dest = OUT / group
        if group in POSE_GROUPS and not (OUT/(group+'_cases.json')).exists():
            call('prepare_survey_pose_intervention')
        if group == 'lego' and not (OUT/'lego_cases.json').exists():
            call('prepare_static_object_survey')
        if group == 'flood_holdout' and not (OUT/'flood_holdout_cases.json').exists():
            call('prepare_survey_flood_holdout')
        base = dest/'predictions.json'
        if not base.exists():
            call('benchmark_stereo_transport','generate',group)
        def stage(module, outputs, *options):
            if any(not (dest/(name+'.json')).is_file() for name in outputs):
                call(module,*options,group)
        stage('benchmark_survey_depth_prior', ('depth_prior','altitude','median_plane','median_stereo'))
        if group not in POSE_GROUPS:
            stage('benchmark_survey_rectification', ('rectified','dual_match'))
            for size in ('256','native'):
                if any('mvsplat_acid_'+size not in r['artifacts'] for r in read(base)):
                    call('benchmark_survey_mvsplat','--upstream',args.mvsplat,'--size',size,group)
            stage('benchmark_survey_mvsplat', ('mvsplat_acid_native_plane_bg',),
                  '--upstream',args.mvsplat,'--size','native','--composite')
            stage('benchmark_survey_depthsplat', ('depthsplat_dl3dv_256x448','depthsplat_dl3dv_256x448_plane_bg'),
                  '--upstream',args.depthsplat,'--dinov2',args.dinov2)
        if group not in ('fresh','confirmation','lego'):
            if group not in POSE_GROUPS:
                stage('benchmark_survey_interpolation',('rife',))
            stage('benchmark_survey_interpolation',('rife_ground',),'--align')
        stage('benchmark_survey_visibility', tuple('mesh_'+r+'_'+m for r in ('qualified','median_stereo')
                                                  for m in ('hard','blend')))
        stage('benchmark_survey_detail', tuple('detail_'+r+'_'+m for r in ('qualified','median_stereo')
                                              for m in ('fixed','adaptive')))
        stage('benchmark_survey_blending', ('simple_average','reverse_bands','sharpen_surface'))
        stage('benchmark_survey_precision', ('precision_depth','precision_donor'))
        if args.learned:
            stage('benchmark_survey_learned_bands',
                  ('learned_bands_geometry','learned_bands_image','learned_bands_constant'),'apply')
        call('benchmark_stereo_transport','score',group)
        if group == 'lego':
            call('score_survey_static_foreground')
    completed = [g for g in FULL_GROUPS+POSE_GROUPS if (OUT/g/'scores.json').is_file()]
    call('summarize_survey_nvs',*completed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups',nargs='+', choices=FULL_GROUPS+POSE_GROUPS)
    parser.add_argument('--learned',action='store_true',help='Also evaluate previously trained geographic folds')
    parser.add_argument('--packages',type=Path,default=EXTERNAL/'envs/survey_mvsplat_packages')
    parser.add_argument('--mvsplat',type=Path,default=EXTERNAL/'code/mvsplat-01f9a28')
    parser.add_argument('--depthsplat',type=Path,default=EXTERNAL/'code/depthsplat-2dad25a7b9ba7c5537aba08463ef88d8511a06be')
    parser.add_argument('--dinov2',type=Path,default=EXTERNAL/'code/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8')
    execute(parser.parse_args())
