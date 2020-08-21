# Worm2vec with harmonic network
This folder contains project that run Worm2vec with harmonic network.

# Run worm2vec
To train network, Run this code.
```
# train
python run_worm2vec.py path.worm_data=/root/worm2vec/data/varietydata_r36_n5_0.npz nn.n_epochs=3 nn.batch_size=100

# predict
python run_worm2vec.py path.worm_data=/root/worm2vec/data/varietydata_r36_n5_0.npz train_mode=False

```