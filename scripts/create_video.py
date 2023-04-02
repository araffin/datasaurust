import sys

import ffmpeg

# from moviepy.editor import VideoFileClip

input_folder = sys.argv[1]
output_video = sys.argv[2]

ffmpeg.input(f"{input_folder}*.png", pattern_type="glob", framerate=25).output(
    output_video
).run()

# Optional: convert video to gif
# video_clip = VideoFileClip(output_video)
# video_clip.write_gif(f"{Path(output_video).parent}/{Path(output_video).stem}.gif")
