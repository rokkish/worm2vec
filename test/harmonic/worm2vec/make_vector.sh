# dir
#mkdir /root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant2
#mkdir /root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant2/train
#mkdir /root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant2/test

# train data
python run_worm2vec.py path.worm_data="./" path.fixedtestdata="/root/worm2vec/data/variety_data_strict_r36_n100_pt/train/varietydata_r36_n100" path.checkpoint_fullpath="/root/worm2vec/worm2vec/test/harmonic/worm2vec/outputs/2021-01-22/12-42-01/checkpoints/model.ckpt" nn.batch_size=1 path.tensorboard=./ train_mode=False nn.n_negative=70 nn.n_classes=10 exp_name="make_rotation_invariant_vector2" predict.view_anchor=True predict.sprite_img_isSaved=False predict.n_embedding=251416 path.save_vector_path="/root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant2/train/" make_vector_mode=true

# test data
python run_worm2vec.py path.worm_data="./" path.fixedtestdata="/root/worm2vec/data/variety_data_strict_r36_n100_pt/test/varietydata_r36_n100" path.checkpoint_fullpath="/root/worm2vec/worm2vec/test/harmonic/worm2vec/outputs/2021-01-22/12-42-01/checkpoints/model.ckpt" nn.batch_size=1 path.tensorboard=./ train_mode=False nn.n_negative=70 nn.n_classes=10 exp_name="make_rotation_invariant_vector2" predict.view_anchor=True predict.sprite_img_isSaved=False predict.n_embedding=196028 path.save_vector_path="/root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant2/test/" make_vector_mode=true