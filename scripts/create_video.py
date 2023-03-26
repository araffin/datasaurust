import sys
from pathlib import Path

import ffmpeg

# from moviepy.editor import VideoFileClip

input_folder = sys.argv[1]
output_video = sys.argv[2]

ffmpeg.input(f"{input_folder}*.png", pattern_type="glob", framerate=25).output(
    output_video
).run()

# video_clip = VideoFileClip(output_video)
# video_clip.write_gif(f"{Path(output_video).parent}/{Path(output_video).stem}.gif")

# Concatenate videos
# input_paths = ["videos/gausian_dog.mp4", "videos/dog_cat.mp4"]

# with open("logs/concat.txt", "w") as f:
#     f.writelines([("file %s\n" % input_path) for input_path in input_paths])

# ffmpeg.input("logs/concat.txt", format="concat", safe=0).output(
#     "logs/videos/merged.mp4", c="copy"
# ).run()
