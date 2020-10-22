# Worm2vec with harmonic network
This folder contains project that run Worm2vec with harmonic network.

# Run worm2vec
To train network, Run this code.
```
# train
python run_worm2vec.py path.worm_data=/root/worm2vec/data/variety_data_r36_n50_np/01.npz nn.n_epochs=3 nn.batch_size=1 train.restart_train=False nn.n_negative=50

# predict
python run_worm2vec.py path.worm_data=/root/worm2vec/data/variety_data_r36_n50_np/01.npz train_mode=False path.tensorboard=./ path.checkpoint_fullpath=/root/worm2vec/worm2vec/test/harmonic/worm2vec/outputs/2020-10-07/04-40-09/checkpoints/model.ckpt nn.batch_size=1 nn.n_negative=50 nn.n_classes=10

# project
tensorboard tensorboard  --logdir . --bind_all --port 8888

```