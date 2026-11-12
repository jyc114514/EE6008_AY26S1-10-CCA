"""Small synthetic correctness checks. No real final labels are read."""
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from compare_core_a import bootstrap_metrics, metrics_and_details, stable_rank


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.columns = np.array([119, 94, 90, 141, 230, 12, 8, 300])
        self.y = np.array([0, 1, 1, 2, 3, 4, 4, 6])
        self.truth = self.columns[self.y]
        self.logits = np.array([
            [8, 6, 5, 4, 3, 2, 1, 0],
            [8, 8, 5, 4, 3, 2, 1, 0],  # tied top1 selects lower column0
            [1, 7, 5, 4, 3, 2, 0, 6],
            [8, 6, 1, 4, 3, 2, 5, 0],
            [0, 6, 5, 7, 3, 2, 1, 4],
            [8, 7, 6, 5, 4, 4, 4, 0],  # tied top5 boundary selects column4
            [8, 7, 6, 5, 1, 4, 3, 2],
            [0, 1, 2, 3, 4, 5, 9, 6],
        ], dtype=float)

    def test_fixed_labels_and_canonical_mapping(self):
        metrics, details, prepared = metrics_and_details(self.truth, self.logits, self.columns)
        rank = stable_rank(self.logits)
        predicted = rank[:, 0]
        np.testing.assert_array_equal(prepared["y"], self.y)
        self.assertEqual(int(self.columns[predicted[1]]), 119)
        self.assertEqual(rank[5, :5].tolist(), [0, 1, 2, 3, 4])
        self.assertAlmostEqual(metrics[0], accuracy_score(self.y, predicted))
        self.assertAlmostEqual(metrics[1], np.mean([y in r[:5] for y, r in zip(self.y, rank)]))
        self.assertAlmostEqual(metrics[2], f1_score(self.y, predicted, labels=np.arange(8), average="macro", zero_division=0))
        precision, recall, f1, support = precision_recall_fscore_support(self.y, predicted, labels=np.arange(8), zero_division=0)
        for i, row in enumerate(details):
            self.assertEqual(row["support"], support[i])
            self.assertAlmostEqual(row["precision"], precision[i])
            self.assertAlmostEqual(row["recall"], recall[i])
            self.assertAlmostEqual(row["f1"], f1[i])
        self.assertEqual(details[5]["support"], 0)
        self.assertEqual(details[5]["f1"], 0)

    def test_shared_paired_bootstrap_and_fixed_class_f1(self):
        _, _, a = metrics_and_details(self.truth, self.logits, self.columns)
        shifted = np.roll(self.logits, 1, axis=1)
        _, _, b = metrics_and_details(self.truth, shifted, self.columns)
        draws = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [7, 6, 6, 4, 3, 3, 1, 0],
        ])
        result = bootstrap_metrics({"a": a, "same_a": a, "b": b}, draws, 8)
        np.testing.assert_array_equal(result["a"] - result["same_a"], np.zeros((4, 3)))
        for i, draw in enumerate(draws):
            for key, data in (("a", a), ("b", b)):
                expected = [
                    accuracy_score(data["y"][draw], data["pred"][draw]),
                    data["top5"][draw].mean(),
                    f1_score(data["y"][draw], data["pred"][draw], labels=np.arange(8), average="macro", zero_division=0),
                ]
                np.testing.assert_allclose(result[key][i], expected, atol=1e-14)
        expected_delta = np.asarray([
            accuracy_score(a["y"][draw], a["pred"][draw]) - accuracy_score(b["y"][draw], b["pred"][draw])
            for draw in draws
        ])
        np.testing.assert_allclose(result["a"][:, 0] - result["b"][:, 0], expected_delta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
