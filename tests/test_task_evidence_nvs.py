import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from task_evidence_nvs.annotation import rebuild_grounded_text, review_generated_view
from task_evidence_nvs.conditions import build_evidence_condition
from task_evidence_nvs.geometry import Camera, PointCloud, orbit_camera
from task_evidence_nvs.scene import SceneBuilder
from task_evidence_nvs.selection import CandidateView, select_candidates
from task_evidence_nvs.task import parse_task


class TaskEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.task = parse_task("检测道路附近的车辆和行人，适应斜视和远距离观测")
        self.camera = Camera(
            np.array([[48, 0, 32], [0, 48, 24], [0, 0, 1]], dtype=np.float32),
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]], dtype=np.float32),
            64,
            48,
        )
        points = PointCloud(
            np.array([[-0.5, 0, 2], [-0.3, 0, 2], [0.5, 0, 2.5], [0.5, 0.2, 2.5]], dtype=np.float32),
            np.ones((4, 3), dtype=np.float32) * 0.5,
            np.ones(4, dtype=np.float32),
            ("car_01", "car_01", "person_01", "person_01"),
        )
        references = [{
            "image_id": "ref",
            "width": 64,
            "height": 48,
            "instances": [
                {"instance_id": "car_01", "category": "car", "bbox": [20, 18, 34, 30]},
                {"instance_id": "person_01", "category": "person", "bbox": [36, 18, 44, 34]},
            ],
        }]
        self.scene = SceneBuilder("scene", self.task).build(references, points)

    def test_parser_extracts_structured_requirements(self):
        self.assertIn("vehicle", self.task.categories)
        self.assertIn("road", self.task.categories)
        self.assertIn("person", self.task.categories)
        self.assertIn("oblique", self.task.view_conditions)
        self.assertIn("near_road", self.task.relations)
        self.assertGreater(self.task.relevance("car"), 0.0)

    def test_projection_condition_is_task_weighted_and_not_label(self):
        condition = build_evidence_condition(self.scene, self.task, self.camera, view_id="v0", context_radius=1)
        self.assertEqual(condition.rgb.shape, (48, 64, 3))
        self.assertGreater(float(condition.task_attention.sum()), 0.0)
        self.assertFalse(condition.metadata["is_target_label"])
        self.assertFalse(condition.metadata["is_generation_reliability"])

    def test_orbit_and_selection_respect_budget(self):
        moved = orbit_camera(self.camera, yaw_deg=10)
        self.assertFalse(np.allclose(moved.world_to_camera, self.camera.world_to_camera))
        candidates = [
            CandidateView("a", {}, 0.9, 0.9, feature=(0,)),
            CandidateView("b", {}, 0.2, 1.0, feature=(1,)),
            CandidateView("c", {}, 0.8, 0.8, feature=(2,)),
        ]
        result = select_candidates(candidates, max_generation_cost=2, max_annotation_cost=2, reliability_threshold=0.5)
        self.assertEqual([item.candidate_id for item in result.selected], ["a", "c"])
        self.assertEqual(result.generation_cost, 2)
        self.assertTrue(any(item["candidate_id"] == "b" for item in result.rejected))

    def test_review_rebuilds_text_without_inheriting_unconfirmed_boxes(self):
        annotation = review_generated_view(
            "v0",
            [{"object_id": "proposal-1", "category": "car", "bbox": [1, 2, 8, 10]}],
            [{"object_id": "proposal-1", "category": "car", "bbox": [2, 3, 9, 11], "status": "preserved", "source_instance_id": "car_01"}],
        )
        text = rebuild_grounded_text(annotation, self.task)
        self.assertEqual(text["grounded_object_ids"], ["proposal-1"])
        self.assertIn("car", text["caption"])
        self.assertEqual(annotation.objects[0].bbox.to_list(), [2.0, 3.0, 9.0, 11.0])

    def test_review_accepts_generator_confirmations_and_tracks_new_objects(self):
        confirmations = (item for item in [
            {"object_id": "new-1", "category": "car", "bbox": [1, 1, 4, 4], "status": "generated_new"},
        ])
        annotation = review_generated_view("v1", [], confirmations)
        self.assertEqual(len(annotation.usable_objects), 1)
        self.assertEqual(annotation.objects[0].status, "generated_new")

    def test_cli_demo_manifest_can_be_processed(self):
        from tools.build_task_evidence_conditions import main
        manifest = Path("configs/task_evidence_demo.json")
        with tempfile.TemporaryDirectory() as directory:
            import sys
            old = sys.argv
            sys.argv = ["build_task_evidence_conditions", "--manifest", str(manifest), "--output", directory,
                        "--max-generation-cost", "2", "--max-annotation-cost", "2"]
            try:
                self.assertEqual(main(), 0)
            finally:
                sys.argv = old
            output = Path(directory)
            self.assertTrue((output / "manifest.json").exists())
            result = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], "task-evidence-conditions-v1")
            self.assertGreater(len(result["candidates"]), 0)


if __name__ == "__main__":
    unittest.main()
