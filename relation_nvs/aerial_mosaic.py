"""Two-source rendering for downward-looking cameras above a horizontal reference plane.

Camera poses are OpenCV camera-to-world in an ENU frame. Only camera geometry and
the two observed RGB images are needed. Synthesized context has provenance 0.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def plane_projection(camera):
    corners = [[x, y, 1.] for x in (0, camera.width - 1) for y in (0, camera.height - 1)]
    rays = np.linalg.solve(camera.intrinsics, np.asarray(corners).T)
    if camera.camera_to_world[2, 3] <= 0 or np.any(camera.camera_to_world[2, :3] @ rays >= -1e-6):
        raise ValueError('Ground-plane rendering requires all image rays to look down from above z=0')
    inverse = np.linalg.inv(camera.camera_to_world)
    return camera.intrinsics @ inverse[:3, [0, 1, 3]]


def transform(points, h):
    shape = points.shape
    flat = np.ascontiguousarray(points.reshape(-1, 1, 2), dtype=np.float64)
    return cv2.perspectiveTransform(flat, h).reshape(shape)


def mutual_matches(first, second):
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.02)
    features = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None)
                for im in (first, second)]
    if any(desc is None or len(desc) < 2 for _, desc in features):
        return np.empty((0, 2)), np.empty((0, 2))
    def good(a, b):
        pairs = cv2.BFMatcher().knnMatch(a, b, k=2)
        return {(m.queryIdx, m.trainIdx) for m, n in pairs if m.distance < .78*n.distance}
    da, db = [d for _, d in features]
    matches = sorted(good(da, db) & {(j, i) for i, j in good(db, da)})
    return tuple(np.asarray([features[k][0][ij[k]].pt for ij in matches], dtype=float).reshape(-1, 2)
                 for k in (0, 1))


def fit_alignment(points, match, nominal):
    """Choose complexity on spatially withheld source-to-source correspondences."""
    if len(points) < 12:
        return nominal, np.empty((0, 2)), np.empty((0, 2)), 'telemetry'
    fold = ((points[:, 0]//64 + points[:, 1]//64) % 2).astype(bool)
    if min(fold.sum(), (~fold).sum()) < 5:
        return nominal, np.empty((0, 2)), np.empty((0, 2)), 'telemetry'
    projected = transform(points, nominal)
    candidates = [('telemetry', nominal)]
    fit = ~fold
    correction, _ = cv2.estimateAffinePartial2D(projected[fit], match[fit], ransacReprojThreshold=3)
    if correction is not None:
        candidates.append(('similarity', np.vstack([correction, [0, 0, 1]]) @ nominal))
    h, _ = cv2.findHomography(points[fit], match[fit], cv2.RANSAC, 3)
    if h is not None and abs(np.linalg.det(h)) > 1e-8:
        candidates.append(('homography', h))
    def loss(item):
        err = np.linalg.norm(transform(points[fold], item[1]) - match[fold], axis=1)
        return np.minimum(err, 10).mean()
    name, h = min(candidates, key=loss)
    delta = match - transform(points, h)
    keep = np.linalg.norm(delta, axis=1) < 6
    if keep[fold].sum() < 4:
        return nominal, np.empty((0, 2)), np.empty((0, 2)), 'telemetry'
    return h, points[keep], delta[keep], name


@dataclass
class AerialMosaic:
    images: tuple
    cameras: tuple
    planes: tuple
    alignment: np.ndarray
    ties: np.ndarray
    residuals: np.ndarray
    model: str

    @classmethod
    def from_views(cls, images, cameras, estimate_alignment=False):
        if len(images) != 2 or len(cameras) != 2:
            raise ValueError('Exactly two source observations required')
        for im, c in zip(images, cameras):
            if im.shape != (c.height, c.width, 3) or im.dtype != np.uint8:
                raise ValueError('Source RGB does not match camera calibration')
        planes = tuple(map(plane_projection, cameras))
        x, y = mutual_matches(*images) if estimate_alignment else (np.empty((0, 2)), np.empty((0, 2)))
        alignment, ties, residuals, model = fit_alignment(x, y, planes[1] @ np.linalg.inv(planes[0]))
        return cls(tuple(images), tuple(cameras), planes, alignment, ties, residuals, model)

    def render(self, target, local=True, texture=True, choose_source=True, registered=False, bounded=True):
        yy, xx = np.indices((target.height, target.width), dtype=np.float32)
        grid = np.stack((xx, yy), axis=-1)
        ground = transform(grid, np.linalg.inv(plane_projection(target)))
        poses = [c.camera_to_world[:3, 3] for c in self.cameras]
        d = np.array([np.linalg.norm(target.camera_to_world[:3, 3]-p) for p in poses])
        nominal = [transform(ground, plane) for plane in self.planes]
        # Preserve target-camera identity at either source pose; bound extrapolation.
        weight = d[0] / max(d.sum(), 1e-9)
        uv = [v.copy() for v in nominal]
        if registered:
            mapped = transform(nominal[0], self.alignment)
            inverse = transform(nominal[1], np.linalg.inv(self.alignment))
            uv = [nominal[0] + weight*(inverse-nominal[0]),
                  nominal[1] + (1-weight)*(mapped-nominal[1])]
        if registered and local and len(self.ties) >= 8:
            distance, idx = cKDTree(self.ties).query(uv[0].reshape(-1, 2), k=4)
            weights = 1/(1+distance)**2
            delta = (weights[..., None]*self.residuals[idx]).sum(axis=1)/weights.sum(axis=1)[:, None]
            delta *= np.exp(-distance[:, :1]/24)
            delta = delta.reshape(grid.shape)
            # Express secondary-source residuals in the primary image frame.
            uv[0] += weight*(transform(mapped-delta, np.linalg.inv(self.alignment))-nominal[0])
            uv[1] += (1-weight)*delta
        if bounded:
            # Georeference corrections propagate only near source correspondences.
            # Avoid distorting distant holes or forcing an unsupported global extrapolation.
            if registered and len(self.ties) >= 8:
                distance, _ = cKDTree(self.ties).query(nominal[0].reshape(-1, 2))
                confidence = np.exp(-(distance.reshape(grid.shape[:2])/32)**2)
                for j in range(2):
                    shift = uv[j]-nominal[j]
                    confidence_j = confidence / np.maximum(1, np.linalg.norm(shift,axis=2)/24)
                    uv[j] = nominal[j]+confidence_j[...,None]*shift
            else:
                uv = nominal
        coords = [v.astype(np.float32) for v in uv]
        colors, margins = [], []
        for image, xy in zip(self.images, coords):
            x, y = xy.transpose(2, 0, 1)
            margins.append(np.minimum.reduce((x, y, image.shape[1]-1-x, image.shape[0]-1-y)))
            colors.append(cv2.remap(image, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))
        margin = np.stack(margins)
        valid = margin >= 0
        support = valid.any(axis=0)
        # A single donor prevents double edges; prefer the more interior, nearer observation.
        border_weight = np.minimum(margin, 16)/16
        quality = border_weight - (d/max(d.sum(), 1e-9))[:, None, None]
        donor = np.argmax(np.where(valid, quality, -np.inf), axis=0)
        donor = np.where(support, donor, np.argmax(margin, axis=0))
        result = np.where((donor == 0)[..., None], colors[0], colors[1])
        if not choose_source:
            both = valid.all(axis=0)
            result[both] = np.rint((colors[0][both].astype(float)+colors[1][both])/2).astype(np.uint8)
        if not texture:
            if support.any():
                indices = distance_transform_edt(~support, return_distances=False, return_indices=True)
                result = result[tuple(indices)]
            else:
                result = np.zeros_like(result)
        provenance = np.where(support, donor+1, 0).astype(np.uint8)
        return result, provenance, coords


def nadir_camera(camera, dji_roll=None):
    """Resolve the half-turn reported by this DJI survey near the nadir singularity."""
    from dataclasses import replace
    if dji_roll is None or abs(dji_roll) <= 90 or camera.camera_to_world[2, 2] > -np.cos(np.deg2rad(2)):
        return camera
    pose = camera.camera_to_world.copy()
    pose[:3, :2] *= -1
    return replace(camera, camera_to_world=pose)


@dataclass
class SurveyCalibration:
    origin: np.ndarray
    correction: np.ndarray
    inliers: int
    scatter_m: float | None

    @classmethod
    def estimate(cls, images, cameras):
        """Fit a horizontal similarity of camera baselines from two source RGB observations."""
        model = AerialMosaic.from_views(images, cameras, estimate_alignment=True)
        origin, second = [c.camera_to_world[:2, 3] for c in cameras]
        baseline = second - origin
        correction = np.eye(2)
        count = len(model.ties)
        scatter = None
        if count >= 12 and np.linalg.norm(baseline) > 1:
            matches = transform(model.ties, model.alignment) + model.residuals
            world = [transform(p, np.linalg.inv(h)) for p, h in zip((model.ties, matches), model.planes)]
            offsets = world[0] - world[1]
            offset = np.median(offsets, axis=0)
            scatter = float(np.median(np.linalg.norm(offsets-offset, axis=1)))
            factor = complex(*(baseline + offset)) / complex(*baseline)
            if scatter < .1 * np.linalg.norm(baseline) and .5 < abs(factor) < 2:
                correction = np.array([[factor.real, -factor.imag], [factor.imag, factor.real]])
        return cls(origin, correction, count, scatter)

    def apply(self, camera):
        from dataclasses import replace
        pose = camera.camera_to_world.copy()
        pose[:2, 3] = self.origin + self.correction @ (pose[:2, 3] - self.origin)
        return replace(camera, camera_to_world=pose)
