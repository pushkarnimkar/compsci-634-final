# Weighted k-Means coresets for active learning

This repository contains my geometric algorithms (Compsci 634, Spring 2022) capstone project. This project proposes a batch-mode active learning method using weighted k-means coresets.

# Installation

Install the following dependencies:

```
$ pip install numpy pandas matplotlib modAL 
```

In addition, this uses [pushkarnimkar/zalanborsos-coresets](https://github.com/pushkarnimkar/zalanborsos-coresets), a fork of [zalanborsos/coresets](https://github.com/zalanborsos/coresets) with minor modifications for coreset implementation. To add it to the `PYTHONPATH`:

```
$ export PYTHONPATH="$PYTHONPATH:/path/to/pushkarnimkar/zalanborsos-coresets"
```

# Repository Organization

* `CS634GeomALProp.pdf`: project report
* `datagen.py`: generator for synthetic dataset
* `stub.py`: executable file - entry point

# Running the script

Execute the script using `python stub.py` with no arguments
