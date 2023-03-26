# DatasauRust

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


Note: The original datasets and python code comes from http://www.autodeskresearch.com/papers/samestats
