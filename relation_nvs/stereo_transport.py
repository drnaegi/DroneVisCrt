"""Transport reciprocal source matches through survey cameras and compare residual hypotheses."""
import cv2
import numpy as np

from .aerial_mosaic import AerialMosaic, plane_projection, transform


def grid_for(camera):
    yy, xx = np.indices((camera.height, camera.width), dtype=np.float32)
    return np.stack((xx, yy), axis=-1)


def sample(field, xy):
    return cv2.remap(field, xy.astype(np.float32), None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def project(xyz, camera):
    local = (xyz - camera.camera_to_world[:3, 3]) @ camera.camera_to_world[:3, :3]
    uvz = local @ camera.intrinsics.T
    return uvz[..., :2] / np.maximum(uvz[..., 2:], 1e-5)


def log_depth_sensitivity(xyz, source, target):
    """Pixel displacement per unit log axial depth, with the source ray held fixed."""
    basis = target.camera_to_world[:3,:3] @ target.intrinsics.T
    p = (xyz-target.camera_to_world[:3,3]) @ basis
    dz = (xyz-source.camera_to_world[:3,3]) @ basis
    z = np.maximum(p[...,2:], 1e-5)
    return (dz[...,:2] - p[...,:2] / z * dz[...,2:]) / z


def precision_depth(prior, depth, valid, reliable, trust, cycle, reprojection, rays, source, other):
    # The median scale fixes bias. Absolute residuals give the normal-consistent
    # scale of the source-only prior, including shape and stereo measurement errors.
    reliable = reliable & valid
    if np.any(reliable):
        residual = np.log(prior[reliable]/depth[reliable])
        sigma = max(float(1.4826*np.median(np.abs(residual))), 1e-4)
    else:
        sigma = np.log(4.)
    safe_depth = np.where(valid,depth,prior)
    xyz = source.camera_to_world[:3,3] + safe_depth[...,None]*rays
    derivative = log_depth_sensitivity(xyz,source,other)
    pixel_variance = 1. + cycle**2 + reprojection**2
    precision = trust * valid * np.sum(derivative**2,axis=-1) / pixel_variance
    posterior_var = 1. / (1./sigma**2 + precision)
    fraction = posterior_var * precision
    fused = prior * np.exp(fraction*np.log(safe_depth/prior))
    return fused, posterior_var, sigma


def select_depth_prior(plane, stereo, mono, reliable):
    """Select a depth prior on spatially held-out source matches, in log-depth units."""
    yy, xx = np.indices(stereo.shape)
    mono_ok = np.isfinite(mono) & (mono > 0)
    ratio = np.where(mono_ok, mono / plane, 1.)
    observed = reliable & mono_ok & (xx % 4 == 0) & (yy % 4 == 0)
    tiles = (xx // 32 + yy // 32) % 2
    fit, check = observed & (tiles == 0), observed & (tiles == 1)
    if min(fit.sum(), check.sum()) < 64:
        return plane, 'plane', {}
    # A shared altitude offset competes with an image-dependent shape estimate.
    y = np.log(np.where(reliable, stereo, plane) / plane)
    x = np.log(ratio)
    center = np.median(x[fit])
    x = np.where(mono_ok, x - center, 0.)
    constant = np.median(y[fit])
    design = np.column_stack((np.ones(fit.sum()), x[fit]))
    coeff = np.array([constant, 1.])
    for _ in range(5):
        error = y[fit] - design @ coeff
        w = np.sqrt(np.minimum(1., .03 / np.maximum(np.abs(error), 1e-7)))
        coeff = np.linalg.lstsq(design * w[:, None], y[fit] * w, rcond=None)[0]
        coeff[1] = np.clip(coeff[1], 0., 2.)
        coeff[0] = np.median(y[fit] - coeff[1]*x[fit])
    offsets = {'plane': np.zeros_like(plane), 'altitude': np.full_like(plane, constant),
               'mono_shape': np.clip(coeff[0] + coeff[1]*x, np.log(.15), np.log(4.))}
    errors = {name: float(np.mean(np.minimum(.25, np.abs(pred[check] - y[check]))))
              for name, pred in offsets.items()}
    winner = min(errors, key=errors.get)
    # In low-parallax regions the stereo shape itself is noisy. Demand a margin
    # in relative depth before extending an image-based prior beyond the overlap.
    if errors['plane'] - errors[winner] < .005:
        winner = 'plane'
    elif winner == 'mono_shape' and errors['altitude'] - errors[winner] < .005:
        winner = 'altitude'
    return plane * np.exp(offsets[winner]), winner, errors


class SurveyStereo:
    def __init__(self, images, cameras, flows, monocular=None, prior_rule='qualified'):
        self.plane = AerialMosaic.from_views(images, cameras)
        self.images, self.cameras = images, cameras
        self.geometry = []
        self.prior_fits = []
        self.depth_variance = []
        for j, cam in enumerate(cameras):
            other = cameras[1-j]
            uv = grid_for(cam)
            match = uv + flows[j]
            backward = sample(flows[1-j], match)
            error = np.linalg.norm(flows[j] + backward, axis=-1)
            inside = ((match >= 0) & (match <= [other.width-1, other.height-1])).all(axis=2)
            trust = inside * np.exp(-.5*(error/2.)**2)
            # Solve the closest distance between rays expressed in world coordinates.
            def rays(x, c):
                pixels = np.dstack((x, np.ones(x.shape[:2])))
                return pixels @ np.linalg.inv(c.intrinsics).T @ c.camera_to_world[:3,:3].T
            r, q = rays(uv, cam), rays(match, other)
            q /= np.linalg.norm(q,axis=2,keepdims=True)
            perpendicular = r - (r*q).sum(axis=2,keepdims=True)*q
            base = other.camera_to_world[:3,3]-cam.camera_to_world[:3,3]
            depth = (perpendicular @ base) / np.maximum((perpendicular**2).sum(axis=2),1e-9)
            plane_depth = -cam.camera_to_world[2,3]/r[...,2]
            valid = (depth > .15*plane_depth) & (depth < 4.*plane_depth)
            weight = trust * valid
            prior = plane_depth
            if monocular is not None:
                raw_xyz = cam.camera_to_world[:3,3] + depth[...,None]*r
                reprojection_error = np.linalg.norm(match - project(raw_xyz, other), axis=-1)
                reliable = valid & (trust > .7) & (reprojection_error < 2.) & (np.linalg.norm(perpendicular, axis=2) > .02)
                mono = np.asarray(monocular[j])
                mono_valid = np.isfinite(mono) & (mono > 0)
                if prior_rule in ('source_fit','rays_rescaled'):
                    from .survey_pointmap_fit import fit_pointmap_depth
                    prior,chosen,losses=fit_pointmap_depth(plane_depth,depth,reliable,mono[0],mono[1],
                                                          scale_only=prior_rule=='rays_rescaled')
                elif prior_rule == 'qualified':
                    prior, chosen, losses = select_depth_prior(plane_depth, depth, mono, reliable)
                elif prior_rule == 'metric_pair':
                    prior = np.where(mono_valid,np.clip(mono,.15*plane_depth,4.*plane_depth),plane_depth)
                    chosen,losses = prior_rule,{}
                elif prior_rule in ('median_stereo', 'median_plane', 'altitude','precision'):
                    anchors = mono_valid & reliable
                    if prior_rule == 'altitude':
                        factor = np.median(depth[anchors] / plane_depth[anchors]) if anchors.any() else 1.
                        prior = factor * plane_depth
                    else:
                        measured = prior_rule in ('median_stereo','precision') and anchors.any()
                        mask = anchors if measured else mono_valid
                        fixed = depth if measured else plane_depth
                        factor = np.median(fixed[mask] / mono[mask]) if mask.any() else 1.
                        prior = np.where(mono_valid, np.clip(mono * factor, .15*plane_depth, 4.*plane_depth), plane_depth)
                    chosen, losses = prior_rule, {}
                else:
                    raise ValueError(prior_rule)
                self.prior_fits.append(dict(model=chosen, errors=losses))
            stabilized = prior + weight*(depth-prior)
            if prior_rule == 'precision':
                if monocular is None:
                    raise ValueError('Precision fusion requires monocular priors')
                stabilized, variance, sigma = precision_depth(prior,depth,valid,anchors,trust,
                                                              error,reprojection_error,r,cam,other)
                self.depth_variance.append(variance)
                self.prior_fits[-1]['prior_log_depth_sigma'] = sigma
            xyz = cam.camera_to_world[:3,3] + stabilized[...,None]*r
            residual = match-project(xyz,other)
            self.geometry.append((uv, xyz, residual * trust[...,None], trust, match))

    def render(self, target, mode='transport', *, precision_donor=False):
        back, prov, nominal = self.plane.render(target)
        if mode == 'plane':
            return back, prov
        qualities, colors = [], []
        for j,(cam, (uv, xyz, residual, trust, match)) in enumerate(zip(self.cameras,self.geometry)):
            other = self.cameras[1-j]
            base = other.camera_to_world[:3,3]-cam.camera_to_world[:3,3]
            delta = target.camera_to_world[:3,3]-cam.camera_to_world[:3,3]
            fraction = np.clip(np.dot(delta,base)/max(np.dot(base,base),1e-9),-1,2)
            pt = project(xyz,target)
            # Transfer the unmodelled endpoint displacement through the other camera's
            # ground homography, in target pixel units. Exact at either endpoint.
            shift_map = plane_projection(target) @ np.linalg.inv(plane_projection(other))
            if mode == 'transport':
                pt += fraction*(transform(match,shift_map)-transform(match-residual,shift_map))
            elif mode == 'flow_plane':
                source_in_target = transform(uv,plane_projection(target)@np.linalg.inv(plane_projection(cam)))
                pt = source_in_target + fraction*trust[...,None]*(transform(match,shift_map)-source_in_target)
            elif mode != 'stereo':
                raise ValueError(mode)
            # Iteratively invert the target warp, using the plane warp for initialization.
            inv = plane_projection(cam)@np.linalg.inv(plane_projection(target))
            movement = (transform(pt,inv)-uv).astype(np.float32)
            xy = nominal[j].copy()
            for _ in range(5):
                xy = nominal[j]-sample(movement,xy)
            roundtrip = np.linalg.norm(xy+sample(movement,xy)-nominal[j],axis=2)
            margin = np.minimum.reduce([xy[...,0],xy[...,1],cam.width-1-xy[...,0],cam.height-1-xy[...,1]])
            quality = np.minimum(margin,16)/16 - np.linalg.norm(delta)/max(
                sum(np.linalg.norm(target.camera_to_world[:3,3]-c.camera_to_world[:3,3]) for c in self.cameras),1e-9)
            if precision_donor:
                if len(self.depth_variance) != len(self.cameras):
                    raise ValueError('Donor uncertainty requires the precision depth rule')
                derivative = log_depth_sensitivity(xyz,cam,target)
                variance = self.depth_variance[j] * np.sum(derivative**2,axis=-1)
                quality -= .5*np.log1p(sample(variance,xy))
            qualities.append(np.where((margin>=0)&(roundtrip<2),quality,-np.inf))
            colors.append(sample(self.images[j],xy))
        best = np.argmax(qualities,axis=0)
        good = np.isfinite(np.max(qualities,axis=0))
        result = np.where((best==0)[...,None],colors[0],colors[1])
        return np.where(good[...,None],result,back),np.where(good,best+1,prov).astype(np.uint8)


def rectify_source_pair(images, cameras):
    """Resample each partner into the first camera's ground-plane coordinates."""
    homographies, rectified = [], []
    for j, camera in enumerate(cameras):
        homography = plane_projection(cameras[1-j]) @ np.linalg.inv(plane_projection(camera))
        xy = transform(grid_for(camera), homography).astype(np.float32)
        rectified.append(cv2.remap(images[1-j], xy, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        homographies.append(homography)
    return rectified, homographies


def unrectify_flow(flows, cameras, homographies):
    mapped = []
    for flow, camera, homography in zip(flows, cameras, homographies):
        source = grid_for(camera)
        mapped.append((transform(source + flow, homography) - source).astype(np.float32))
    return np.stack(mapped)


def qualify_flows(images, cameras, candidates):
    """Compare endpoint patches and round trips in original source pixels."""
    chosen, qualities = [], []
    for j, camera in enumerate(cameras):
        grid = grid_for(camera)
        costs = []
        for flows in candidates:
            endpoint = grid + flows[j]
            color = sample(images[1-j], endpoint)
            err = np.abs(images[j].astype(float) - color) / 255.
            patch = cv2.boxFilter(np.minimum(err, .2).mean(axis=-1), -1, (5, 5))
            cycle = np.linalg.norm(flows[j] + sample(flows[1-j], endpoint), axis=-1)
            valid = ((endpoint >= 0) & (endpoint <= [cameras[1-j].width-1, cameras[1-j].height-1])).all(axis=-1)
            costs.append(np.where(valid, patch + .02 * np.minimum(cycle, 10), np.inf))
        # Ties include absence of overlap. The rectified model starts from the
        # physical homography and is the first candidate.
        first = costs[0] <= costs[1]
        chosen.append(np.where(first[...,None], candidates[0][j], candidates[1][j]))
        qualities.append(float(first.mean()))
    return np.stack(chosen), qualities
