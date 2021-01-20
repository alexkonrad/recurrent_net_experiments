# Recurrent Nets

LSTM/GRU-style RNN implemented in PyTorch with PyTorch Lightning for flexibility in logging and analysis. This repo adds some new datasets that are helpful for experiments and research with RNNs.

## Datasets

### Echo Kth Number

For variable-length sequences and some parameter K, this dataset is helpful for training an RNN to memorize the location to echo. It is a very simple dataset that is helpful for debugging.

```
0 1 2 3 4 5 -> 1
9 2 3 9 -> 2
0 7 8 2 2 0 5 3 2 -> 7
```

### Binary Addition

The binary addition task adds 7-bit binary numbers in order to teach the RNN to add and carry the appropriate numbers.

```
11001
00101
-----
11110
```

### Language Modeling

The Shakespeare text file dataset is also included.
