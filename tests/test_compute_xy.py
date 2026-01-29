import unittest
import pandas as pd
import numpy as np
from bounce_detector.pipeline.a_compute_xy import compute_xy_center, compute_xy_velocity, compute_xy_acceleration


class TestComputeXY(unittest.TestCase):
    def setUp(self):
        # Create a simple dataframe simulating a ball moving
        # Frame 0 to 4: moving right (x increases) and down (y increases) constantly
        self.df = pd.DataFrame({
            'ball_center_x': [10, 12, 14, 16, 18],
            'ball_center_y': [100, 102, 104, 106, 108],
            'time': [0.0, 0.1, 0.2, 0.3, 0.4]
        })

    def test_compute_xy_center_smoothing(self):
        # Test smoothing doesn't crash and returns same length
        xw = 2
        yw = 2
        
        # Save original to compare
        original_x = self.df['ball_center_x'].copy()
        
        df_res = compute_xy_center(self.df, x_window=xw, y_window=yw)
        
        # Implementation overwrites the columns
        self.assertIn('ball_center_x', df_res.columns)
        self.assertIn('ball_center_y', df_res.columns)
        self.assertEqual(len(df_res), len(self.df))
        
        # With window 2, it's a moving average. 
        self.assertTrue(np.allclose(df_res['ball_center_x'], original_x, atol=1.0))

    def test_compute_xy_velocity(self):
        # Need smooth columns first (act as if they were computed)
        # In this test we just want to test velocity calc from current column values
        
        # Constant velocity: dx = 2 pixels per frame
        df_res = compute_xy_velocity(self.df, x_window=1, y_window=1)
        
        self.assertIn('vel_x', df_res.columns)
        self.assertIn('vel_y', df_res.columns)
        
        # Check middle values (edges might have boundary effects)
        mid_vel_x = df_res['vel_x'].iloc[2]
        # Implementation is diff per frame (px/frame), not px/second
        self.assertAlmostEqual(mid_vel_x, 2.0, delta=0.1)

    def test_compute_xy_acceleration(self):
        # Constant velocity implies zero acceleration
        self.df['vel_x'] = 2.0
        self.df['vel_y'] = 2.0
        self.df['time'] = [0.0, 0.1, 0.2, 0.3, 0.4]
        
        df_res = compute_xy_acceleration(self.df, x_window=1, y_window=1)
        
        self.assertIn('acc_x', df_res.columns)
        self.assertIn('acc_y', df_res.columns)
        
        # Acceleration should be close to 0
        mid_acc_x = df_res['acc_x'].iloc[2]
        self.assertAlmostEqual(mid_acc_x, 0.0, delta=1.0)

if __name__ == '__main__':
    unittest.main()
