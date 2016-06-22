[WIP] Driver Action in Kaggle
========

Driver action recognition challenge in [Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection).
Implemented by Chainer.
This repository is work in progress.

# Requirements

- [Chainer 1.5+](https://github.com/pfnet/chainer) (Neural network framework)


# Preparation

## prepare data

```
mkdir data
cd data
unzip sample_submission.csv.zip
unzip imgs.zip
unzip driver_imgs_list.csv.zip
cd ..
```

## make train and validation data

```
python scripts/dataset.py
```

## download pre-train model

```
bash scripts/download.sh
```

## calcurate mean

```
python scripts/calc_mean.py
```

# Start training

```
python scripts/train.py
```
