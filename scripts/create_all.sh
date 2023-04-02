#!/bin/bash

# Output folder for videos
video_folder=logs/videos_0
mkdir -p $video_folder
log_folders=(cat dog rabbit)

# Remove old logs
for log_folder in "${log_folders[@]}"
do
    rm -r logs/$log_folder
done

# Create cat animation
cargo run --release -- -p -n 1000000 --decimals 2 --shape cat --allowed-distance 0.1 --log-interval 10000 --gaussian --save-plots

# Create dog animation from last cat frame
cargo run --release -- -p -n 1200000 --decimals 2 --shape dog --allowed-distance 0.1 --log-interval 10000 -d logs/cat/output.csv --save-plots

# # Create rabbit animation from last dog frame
cargo run --release -- -p -n 1200000 --decimals 2 --shape rabbit --allowed-distance 0.1 --log-interval 10000 -d logs/dog/output.csv --save-plots

# Create horse animation from last rabbit frame
# cargo run --release -- -p -n 1000000 --decimals 2 --shape horse --allowed-distance 0.1 --log-interval 10000 -d logs/rabbit/output.csv


# Create videos
video_list=()
for log_folder in "${log_folders[@]}"
do
    echo "Creating video for $log_folder"
    python3 scripts/create_video.py logs/$log_folder/ $video_folder/$log_folder.mp4

    video_list+=($video_folder/$log_folder.mp4)
done

# Concatenate videos
python3 scripts/concat_videos.py -i ${video_list[@]} -o $video_folder/concat.mp4