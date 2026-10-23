import unittest

import numpy as np
import torch

from breastseg.instance_metrics import (
    bilkent_metrics,
    bbbc039_reference_instances,
    classmap_to_instances,
    published_metrics,
    semantic_to_instances,
)
from train_bbbc_instance import three_class_target
from train_bbbc_contact import contact_instances, contact_loss, touching_contact_target


class InstanceMetricContracts(unittest.TestCase):
    def test_bbbc_graph_colours_preserve_touching_instances(self):
        raw = np.zeros((8, 8, 4), np.uint8)
        raw[2:6, 1:4, 2] = 1
        raw[2:6, 4:7, 2] = 2
        labels = bbbc039_reference_instances(raw, min_size=1)
        self.assertEqual(labels.max(), 2)

    def test_three_class_target_separates_graph_colours(self):
        raw = np.zeros((12, 12, 4), np.uint8)
        raw[3:9, 1:6, 2] = 1
        raw[3:9, 6:11, 2] = 2
        instances = bbbc039_reference_instances(raw, min_size=1)
        classes = three_class_target(instances)
        self.assertTrue(np.isin(classes, (0, 1, 2)).all())
        self.assertTrue(np.all(classes[3:9, 5:7] == 2))
        self.assertTrue(np.any(classes[4:8, 2:4] == 1))

    def test_contact_target_marks_only_touching_interface(self):
        instances = np.zeros((12, 12), np.int32)
        instances[3:9, 1:6] = 1
        instances[3:9, 6:11] = 2
        contact = touching_contact_target(instances, radius=0)
        self.assertTrue(contact[:, 5:7].any())
        self.assertFalse(contact[:, :1].any())
        separated = contact_instances(instances > 0, contact, min_size=1)
        self.assertEqual(separated.max(), 2)

    def test_contact_loss_is_finite_for_zero_positive_batch(self):
        logits = torch.zeros((2, 2, 8, 8), requires_grad=True)
        target = torch.zeros_like(logits)
        loss = contact_loss(logits, target, contact_weight=.5)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_perfect_instances(self):
        target = np.zeros((12, 12), np.int32)
        target[1:4, 1:4] = 1
        target[7:10, 7:10] = 2
        result = published_metrics(target, target)
        for key in ("dice1", "dice2", "aji", "dq", "sq", "pq", "caicedo_af1", "symmetric_best_dice"):
            self.assertAlmostEqual(result[key], 1.0)

    def test_empty_prediction(self):
        target = np.zeros((12, 12), np.int32)
        target[1:5, 1:5] = 1
        result = published_metrics(target, np.zeros_like(target))
        for key in ("dice1", "dice2", "aji", "dq", "sq", "pq", "caicedo_af1", "symmetric_best_dice"):
            self.assertEqual(result[key], 0.0)

    def test_aji_penalizes_completely_missed_reference(self):
        target = np.zeros((8, 8), np.int32)
        target[1:3, 1:3] = 1
        target[5:7, 5:7] = 2
        prediction = np.zeros_like(target)
        prediction[1:3, 1:3] = 1
        self.assertAlmostEqual(published_metrics(target, prediction)["aji"], 0.5)

    def test_component_conversion(self):
        binary = np.zeros((20, 20), bool)
        binary[1:6, 1:6] = True
        binary[12:18, 12:18] = True
        labels = semantic_to_instances(binary, min_size=4)
        self.assertEqual(labels.max(), 2)

    def test_boundary_expansion_preserves_instances(self):
        classes = np.zeros((20, 20), np.uint8)
        classes[3:8, 2:7] = 1
        classes[3:8, 8:13] = 1
        classes[3:8, 7] = 2
        interior = classmap_to_instances(classes, expand_boundary=False, min_size=4)
        expanded = classmap_to_instances(classes, expand_boundary=True, min_size=4)
        self.assertEqual(interior.max(), 2)
        self.assertEqual(expanded.max(), 2)
        self.assertGreater((expanded > 0).sum(), (interior > 0).sum())

    def test_bilkent_perfect_and_merged_objects(self):
        target = np.zeros((8, 8), np.int32)
        target[1:4, 1:4] = 1
        target[1:4, 4:7] = 2
        perfect = bilkent_metrics(target, target)
        self.assertEqual(perfect["nucleus_f1"], 1.0)
        self.assertEqual(perfect["pixel_f1"], 1.0)
        merged = np.zeros_like(target)
        merged[target > 0] = 1
        result = bilkent_metrics(target, merged)
        self.assertLess(result["nucleus_f1"], 1.0)
        self.assertEqual(result["n_prediction"], 1)
        self.assertEqual(result["n_reference_pixels"], int((target > 0).sum()))
        self.assertEqual(result["n_prediction_pixels"], int((merged > 0).sum()))


if __name__ == "__main__":
    unittest.main()
