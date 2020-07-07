# Overview

# Environments
Pytorch
TensorboardX

# Preprocess

## **Preprocess** data, and save as torch
生データに前処理（線虫の切り出し）を施し，tensorをバイナリ形式で保存
```
python preprocess.py --process_id 0~3 --save_name processed
```

## **Rename** binary data
日付ごとに別れているデータを統合し，ファイル名に日付を追記
```
python features/rename.py
```

## **Make** distance table
類似度テーブルの作成
max_original個のデータについて，それぞれmax_pair個のデータとのMSEを取得
max_original x max_pair行のデータが得られる．
```
python get_distance_table.py --process_id 0~3 --max_pair 100 --max_original 20000
```

## **Compress** distance table top K
類似度テーブルの圧縮
max_original x max_pair行のデータから，ネガティブサンプルとして誤差が大きいtop Kだけ選出
max_original x K行のデータが得られる．
```
python compress_distance_table.py -K 1~100
```

## **Make** variety_dataset [original, rotation, negative] from compressed distance table
類似度テーブルからネガティブサンプルデータセットの作成
max_original x K行のデータに基づいて，データセットを作成
load_Kは読み込みデータの指定("distance_table_compress_top?")，num_negativeは作成データセットのtopKを指定
出力は(num_rotate + num_negative, 1, 64, 64) x データ数(max_original)
```
python make_variety_dataset.py --load_K 1~100 --num_rotate 1~36 --num_negative 1~5 --save_path varietydata
```

# Main

## **Train** embedding model
```
python train.py --epoch 2 --logdir test_nce --gpu_id 1 --traindir processed/varietydata --model_name test_triplet --model worm2vec_nonseq --loss_function_name TripletMargin --zsize 128
```

## **Embed** image into latent vector
```
python predict.py --logdir test_triplet --gpu_id 2 --traindir processed/varietydata --model_name test_triplet --model worm2vec_nonseq --test_shuffle --zsize 128 --max_predict 3
```

# Vizualize

## **Run** jupyter notebook
```
jupyter notebook --allow-root --ip 0.0.0.0 --port
```
## **Run** tensorboard
```
tensorboard tensorboard --logdir . --bind_all --port
```

## **Analyze** midium layers
```
python analyze.py --logdir test_triplet_z64_data16k --gpu_id 2 --traindir processed/varietydata --model_name test_triplet --model worm2vec_nonseq --zsize 128 --max_analyze 1
```

# Requirements
```
pip install -r ../documents/requirements.txt
```

# Dir
```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```