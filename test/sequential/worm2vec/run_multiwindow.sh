# window in [5, 10, 20]:
#exp_name="fulltarin_25k_w05"
#python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=True train.window_size=5 exp_name=$exp_name predict.single_view_mode=True
#exp_name="fulltarin_25k_w10"
#python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=True train.window_size=10 exp_name=$exp_name predict.single_view_mode=True
exp_name="fulltarin_stimulus_25k_w20"
#python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=True train.window_size=20 exp_name=$exp_name predict.single_view_mode=True
exp_name="fulltarin_nostimulus_19k_w20"
python run_worm2vec.py train.n_samples=190000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=True train.window_size=20 exp_name=$exp_name predict.single_view_mode=True dir.data="/root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant/test/tensor.csv" dir.metadata="/root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant/test/metadata.tsv" dir.test.data="/root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant/train/tensor.csv" dir.test.metadata="/root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant/train/metadata.tsv" dir.test.original="/root/worm2vec/data/alldata_tanimoto/alldata/alldata"
#python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=False train.window_size=40 exp_name=$exp_name
#python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=False train.window_size=50 exp_name=$exp_name
#python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=False train.window_size=60 exp_name=$exp_name
#python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=False train.window_size=70 exp_name=$exp_name