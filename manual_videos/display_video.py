import cv2  # Video player
import random
import numpy as np

# Set the number of trajectories to display (number of files in manual_videos/videos).
num_trajs = 20
trajs = []

# Add the string corresponding to the video file name for each trajectory to trajs.
for i in range(1, num_trajs + 1):
    if len(str(i)) == 1:
        trajs.append("traj_0" + str(i))  # Single digit.
    else:
        trajs.append("traj_" + str(i))  # Double digits.

# Create two copies of trajs. unwatched_trajs all trajectories that have not yet been watched. Every iteration two
# trajectories are chosen at random from unwatched_trajs and then added to watched_trajs. These two trajectories are
# then displayed side-by-side using cv2.
unwatched_trajs = trajs.copy()  # .copy() so these don't point to trajs.
watched_trajs = trajs.copy()

i = 0
while i <= num_trajs / 2:
    traj_1 = random.choice(unwatched_trajs)
    unwatched_trajs.remove(traj_1)
    watched_trajs.append(traj_1)

    # Note that traj_2 is only chosen after traj_1 has been removed from unwatched_trajs to prevent the same trajectory
    # from being chosen twice.
    traj_2 = random.choice(unwatched_trajs)
    unwatched_trajs.remove(traj_2)
    watched_trajs.append(traj_2)

    # Some ungodly slop spewed out by GPT o3. I left some comments but idk what is really going on here.
    while True:
        cap1 = cv2.VideoCapture(f"videos/{traj_1}.mp4")  # This fetches the two trajectories.
        cap2 = cv2.VideoCapture(f"videos/{traj_2}.mp4")

        target_h = 480
        while True:
            ok1, f1 = cap1.read()
            ok2, f2 = cap2.read()
            if not ok1 or not ok2:  # GPT "safety" measure.
                break

            # This allegedly scales the two videos.
            scale1 = target_h / f1.shape[0]
            scale2 = target_h / f2.shape[0]
            f1 = cv2.resize(f1, (int(f1.shape[1] * scale1), target_h))
            f2 = cv2.resize(f2, (int(f2.shape[1] * scale2), target_h))

            cv2.imshow("side-by-side", cv2.hconcat([f1, f2]))  # Display videos.
            cv2.waitKey(30)  # Lower frame rate to around 20 fps.

        # Close both videos once they are done playing. The one which finishes first remains on the final frame till the
        # second also finishes.
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

        # Hook for RLHF trajectories. Doesn't do anything yet beside replay, but this is where you can hook in Nathan
        # to learn WFA.
        ui = input("Did you think the left [1] or right [2] clip was better? If not sure input 0 to replay both: ")
        if ui != "0":
            break  # If no replay requested, break loop and play next pair -- otherwise replay.

    i += 1
