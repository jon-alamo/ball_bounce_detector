import unittest
import pandas as pd
import numpy as np
from bounce_detector.pipeline.h_classify_candidates import extract_features

class TestMLUtils(unittest.TestCase):
    def test_extract_features(self):
        # Test feature extraction robustness
        cols = ['vel_x', 'vel_y', 'acc_x', 'acc_y', 'vel_angle', 'vel_module', 'acc_angle', 'acc_module']
        df = pd.DataFrame(np.random.rand(20, len(cols)), columns=cols)
        
        # Test valid extraction
        features = extract_features(df, candidate_frame=10, window=5)
        # Size: (window*2 + 1) * len(cols) = 11 * 8 = 88
        self.assertEqual(len(features), 88)
        
        # Test boundary (should return None)
        features_edge = extract_features(df, candidate_frame=0, window=5)
        self.assertIsNone(features_edge)

if __name__ == '__main__':
    unittest.main()