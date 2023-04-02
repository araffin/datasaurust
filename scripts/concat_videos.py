import argparse
from pathlib import Path

import ffmpeg

# Parse arguments to concatenate videos
parser = argparse.ArgumentParser()
# List of videos to concatenate
parser.add_argument("-i", "--input-videos", type=str, nargs="+", required=True)
parser.add_argument("-o", "--output-video", type=str, required=True)
args = parser.parse_args()


# Remove the "logs/" prefix from the input paths
input_paths = [
    str(Path(input_path).relative_to("logs")) for input_path in args.input_videos
]

with open("logs/concat.txt", "w") as f:
    f.writelines([("file %s\n" % input_path) for input_path in input_paths])

ffmpeg.input("logs/concat.txt", format="concat", safe=0).output(
    args.output_video, c="copy"
).run()
