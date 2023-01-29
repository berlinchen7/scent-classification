# Scent Classification

## Set up
1. Create a python virtual environment (using, e.g., Anaconda). The code is developed using Python 3.8.12, so a similar Python version is recommended.
2. Inside the virual environment, run the `build.sh` script and specify the OS/resouce type, so for Mac+CPU this would be:
```sh
source build.sh mac-cpu
```
2. Simply run the following to train an epoch of the combined Goodscents+Leffingwell dataset using a GNN:
```sh
python src/gnn_classifier.py
```