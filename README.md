# DatasauRust

Blazingly fast implementation of the [Datasaurus](https://www.autodesk.com/research/publications/same-stats-different-graphs) paper (500x faster than the original): "Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics through Simulated Annealing" by Justin Matejka and George Fitzmaurice.

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

Create video and gif (use `--save-plot`):
```
pip install moviepy ffmpeg-python

python scripts/create_video.py logs/cat/ logs/cat.mp4
```


Note: The original datasets and python code comes from http://www.autodeskresearch.com/papers/samestats
