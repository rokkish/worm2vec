# Sequential model for toy-data
## dataset
[four-shapes](https://www.kaggle.com/smeschke/four-shapes)

## preprocess
### 希望
* データの作成: 二種類の図形の画素平均
* １fileに入出力入れたい
* ndarray (2*X + Y, H, W) = {X: input, Y: output}
* png -> np.arrayに変換してから処理．

### 作成方法の詳細
2種類の図形画像をA，Bとする．AとBの画素平均から遷移画像Tを作成する．画素平均の係数をlambdaとすると，次式で表現される．
```math
T = (A + \lambda * B)/2 for \lambda in range(0, 1.1, 0.1)
```

## train

# Sequential model for worm
