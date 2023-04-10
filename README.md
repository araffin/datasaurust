![CI](https://github.com/araffin/datasaurust/workflows/CI/badge.svg)

# DatasauRust

Blazingly fast implementation of the [Datasaurus](https://www.autodesk.com/research/publications/same-stats-different-graphs) paper (500x faster than the original): "Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics through Simulated Annealing" by Justin Matejka and George Fitzmaurice.


https://user-images.githubusercontent.com/1973948/230972049-adcb8012-f25f-4df4-84ce-aafc7f58f184.mp4




## Usage

To run with plot `-p` (using gnuplot):
```
cargo run --release -- -d data/seed_datasets/Datasaurus_data.csv -p
```

With pre-defined shape:
```
cargo run --release -- -p -n 3000000 --decimals 2 --shape cat --allowed-distance 0.1
```

Starting from Gaussian noise:
```
cargo run --release -- -p -n 3000000 --decimals 2 --shape cat --allowed-distance 0.1 --gaussian
```

## Create videos

Create video and gif (use `--save-plot`):
```
pip install moviepy ffmpeg-python

python scripts/create_video.py logs/cat/ logs/cat.mp4
```

From one shape to another:
```
cargo run --release -- -p -n 2000000 --decimals 1 --shape dog --allowed-distance 0.1 --log-interval 10000 -d logs/gaussian_cat/output.csv --save-plots
```


Note: The original datasets and python code comes from http://www.autodeskresearch.com/papers/samestats
