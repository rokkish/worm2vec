# Worm2Vec
Copyright (c) 2019 hirogithu

## Overview
* Data - 3万枚の線虫の姿勢画像
* Goal - 姿勢空間の作成と可視化
* Model - AutoEncoder + Word2Vec
  - AutoEncoder
    - 潜在表現の生成機構
    - 回転不変な潜在表現を獲得するための機構
  - Word2Vec
    - 時間的に近い姿勢が得られる表現を目指す．
      - 次の瞬間にとりうる姿勢
    

## Goal
* 線虫の姿勢空間から知見獲得を目指す．
* 貢献
  - 回転不変技術の考案

## Task
* 実装
  - [x] Preprocessing
  - [x] Model arch
  - [] Evaluation
  - [] Visualization

* サーベイ
  - [] Word2Vec
  - [] Rotation invariant tech

### history
  * 2019/12/19: Project started
  * 2019/12/20: Outline made
