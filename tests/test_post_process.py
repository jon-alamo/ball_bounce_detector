import unittest
import pandas as pd
from bounce_detector.pipeline.g_post_process import post_process

class TestPostProcess(unittest.TestCase):
    def setUp(self):
        # Create simulation of shots
        # 10 frames total
        data = {
            'ball_center_y': [10] * 10,
            'is_bounce_detected': [0] * 10,
            'category': [''] * 10
        }
        self.df = pd.DataFrame(data)

    def test_serve_logic_no_bounce(self):
        # If Serve and no bounce detected, should pick max height (max y? No, coordinate y is pixels from top. 
        # So "height" in real world is usually min y in pixels if camera is standard, BUT
        # let's look at implementation: max_height_frame = shot_df['ball_center_y'].idxmax()
        # Wait, ball_center_y is pixels from top-left. So larger Y means LOWER in the image (closer to floor).
        # So idxmax() picks the LOWEST point in image (likely the bounce on floor). Correct.
        
        # Set frames 0-4 as Serve
        self.df.loc[0:5, 'category'] = 'Serve'
        self.df.loc[3, 'ball_center_y'] = 100 # Peak (lowest point in image)
        
        df_res = post_process(self.df, shot_bounce=0.5)
        
        # Should have detected bounce at frame 3
        self.assertEqual(df_res.loc[3, 'is_bounce_detected'], 1)
        self.assertEqual(df_res['is_bounce_detected'].sum(), 1)

    def test_serve_logic_with_multiple_bounces(self):
        # If Serve and bounces detected, keep only LAST one
        self.df.loc[0:5, 'category'] = 'Serve'
        self.df.loc[1, 'is_bounce_detected'] = 1
        self.df.loc[3, 'is_bounce_detected'] = 1
        
        df_res = post_process(self.df)
        
        self.assertEqual(df_res.loc[1, 'is_bounce_detected'], 0)
        self.assertEqual(df_res.loc[3, 'is_bounce_detected'], 1)

    def test_other_shot_no_bounce(self):
        # If Other shot and no bounce, use shot_bounce ratio
        self.df.loc[0:4, 'category'] = 'Forehand' # Frames 0,1,2,3. Length 4.
        # shot_bounce = 0.5. Offset = 4 * 0.5 = 2. Start (0) + 2 = 2.
        
        df_res = post_process(self.df, shot_bounce=0.5)
        
        self.assertEqual(df_res.loc[2, 'is_bounce_detected'], 1)

if __name__ == '__main__':
    unittest.main()