# Ball bounce detector
This application consists of a detector for direction changes of a padel ball based on a dataset where the ball has been labeled for each frame. The ball is given in each frame by the coordinates of its center, "x" and "y" in pixels, where "x" is the number of pixels horizontally from the top-left corner of the image to the right, and the "y" axis is the number of pixels from the top-left corner vertically downwards.

The source dataset is located in the "datasets/2022-master-finals-fem" directory and contains 2 files:

- The original dataset with all annotated data "datasets/2022-master-finals-fem/2022-master-finals-fem.csv"
- A sample file created manually to validate the detector result for a series of frames.

The files have the following structure.

## Dataset files

### Original Dataset

File located at "datasets/2022-master-finals-fem/2022-master-finals-fem.csv" with the following columns (marking only those relevant to the task in question):

- Row number -1: frame number
- has_shot: whether a player is making a shot on the ball. It covers from the preparation of the shot to the finish, and the contact between the racket and the ball usually occurs in some intermediate frame.
- category: the category of the shot. Can be "Serve", "Forehand", "Backhand", "Smash", and "Other".
- ball_center_x: "x" coordinate in pixels of the ball center.
- ball_center_y: "y" coordinate in pixels of the ball center.
- serving: initials of the player serving in that game.
- serving_team: the team that is serving: "a" or "b".
- upper_team: the team playing at the top of the image or in the area "farthest" from the camera.
- lower_team: the team playing at the bottom of the image or in the area "closest" to the camera.
- team_shot: the team making the shot if has_shot is 1.
- player_<a/b>_<left/drive>_<x/y/w/h>: bbox of each player (team a or b, position left or drive, coordinates x, y, w, and h of the bbox).
- time: the time in seconds corresponding to each frame.
- shot_result: the result of the shot made if has_shot is 1. Can be 1 if the shot is good and play continues, 0 if the shot is bad and the point ends, 2 if the shot is a winner and the point also ends.


### Bounces Sample File

File located at "datasets/2022-master-finals-fem/2022-master-finals-fem-bounces-sample.csv" which contains in which frames ball bounces occur. Contains a single column called "bounce_frame" and indicates that there is a bounce in that frame. A bounce is considered any change of direction suffered by the ball after impacting any surface. It can be a player's racket during a shot, the bounce on the floor, glass, or fences. This type of movement must be distinguished from the change of direction suffered by the effect of gravity itself.


### Application Objective

The objective of the application is to maximize an objective function, which I have defined as:

FO = TP / (TP + FP + FN) where:

TP: true positives -> bounce detected at the right moment
FP: false positives -> bounce detected when there is no bounce
FN: false negatives -> real bounce not detected

In this way, the function will be equal to 1 only when all and only real bounces are detected.


### Considerations

The ball position annotations are made manually, so there is inherent noise in the ball trajectory caused by small errors when entering the data. Furthermore, depending on the direction the ball is going, the bounce becomes much more evident or less so. For example, if the ball is going in a direction perpendicular to the screen itself, its center will remain almost motionless, and upon rebounding, an excessive change of direction may not be noticed either. However, with the human eye, more than 95% of bounces could be detected at any moment, so I believe there must be some way to detect with similar precision in which frames the ball actually bounces.

In addition, a margin of frames can be accepted in which a "TP" would be considered; for example, if the bounce is detected and is 1 frame away from the real bounce, it would be considered a "TP".
