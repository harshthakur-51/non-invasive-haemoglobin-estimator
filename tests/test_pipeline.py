import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from haemoglobin_estimator.cli import evaluate
from haemoglobin_estimator.features import feature_vector


class PipelineTests(unittest.TestCase):
    def test_feature_vector_has_interpretable_inputs(self):
        self.assertEqual(len(feature_vector(160, 120, 100, 25, "female")), 8)

    def test_synthetic_model_has_low_error(self):
        self.assertLess(evaluate()["mae"], 1.25)


if __name__ == "__main__":
    unittest.main()
