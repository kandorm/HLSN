# HLSN

Implementation of "Balance the Labels: Hierarchical Label Structured Network for Dialogue Act Recognition" (IJCNN 2021). Some code in this repository is based on the excellent open-source project https://github.com/iYUYUE/CIKM-2019-CDA.

## Requirements

* Python 3 with PyTorch, pandas and sklearn

## Data

* [Mastodon](https://github.com/cerisara/DialogSentimentMastodon)
* [Dialogbank](https://dialogbank.uvt.nl/)

## Train

```
python src/run.py train --ld 0.3 --lr 0.002 --lstm_hidden 400
```


## Test

```
python src/run.py test --model ./model/geqybwhcae/ --model_file 5 --lstm_hidden 400
```


### Tune

```
python src/run.py tune
```
