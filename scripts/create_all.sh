#!/bin/bash

# Output folder for videos
video_folder=logs/videos_0
mkdir -p $video_folder
log_folders=(cat dog whale rabbit)

# Remove old logs
for log_folder in "${log_folders[@]}"
do
    rm -r logs/$log_folder
done
rm logs/videos_0/*

default_args="-p --decimals 2 --allowed-distance 0.1 --log-interval 10000 --save-plots"

# Create cat animation
cargo run --release -- -n 1000000 --shape cat --gaussian --seed 107 $default_args

# Create dog animation from last cat frame
cargo run --release -- -n 1400000 --shape dog -d logs/cat/output.csv $default_args

# # Create whale animation
cargo run --release -- -n 4000000 --max-temperature 0.99 --shape whale -d logs/dog/output.csv $default_args

# Create rabiit animation
cargo run --release -- -n 2000000 --max-temperature 0.6 --shape rabbit -d logs/whale/output.csv $default_args


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

# Launch mpv
mpv $video_folder/concat.mp4