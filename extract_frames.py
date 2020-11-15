import glob
import os
from natural_rl_environment.imgsource import video_to_frames


g = '/home/rakelly/garage/distractors/videos/*.mp4'
output = '/home/rakelly/garage/distractors/images'
files = glob.glob(os.path.expanduser(g))
video_to_frames(files, output)
