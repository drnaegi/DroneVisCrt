"""Task-semantic, evidence-constrained multi-view augmentation.

The package keeps scene evidence and generated supervision separate.  Geometry
can propose a label, but only a reviewed generated view can turn that proposal
into training supervision.
"""

from .annotation import GeneratedViewAnnotation, rebuild_grounded_text, review_generated_view
from .conditions import EvidenceCondition, build_evidence_condition
from .geometry import Camera, PointCloud, project_point_cloud, orbit_camera
from .scene import Box, InstanceObservation, SceneBuilder, SceneInstance, SceneRepresentation
from .selection import CandidateView, SelectionResult, select_candidates
from .task import EvidenceRequirement, TaskSpec, parse_task

__all__ = [
    "Camera",
    "Box",
    "CandidateView",
    "EvidenceCondition",
    "EvidenceRequirement",
    "GeneratedViewAnnotation",
    "PointCloud",
    "InstanceObservation",
    "SceneBuilder",
    "SceneInstance",
    "SceneRepresentation",
    "SelectionResult",
    "TaskSpec",
    "build_evidence_condition",
    "orbit_camera",
    "parse_task",
    "project_point_cloud",
    "rebuild_grounded_text",
    "review_generated_view",
    "select_candidates",
]
