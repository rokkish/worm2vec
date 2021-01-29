# window in [5, 10, 20]:
exp_name="fulltarin_25k_w05"
python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=True train.window_size=5 exp_name=$exp_name predict.single_view_mode=True
exp_name="fulltarin_25k_w10"
python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=True train.window_size=10 exp_name=$exp_name predict.single_view_mode=True
exp_name="fulltarin_25k_w20"
python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=True train.window_size=20 exp_name=$exp_name predict.single_view_mode=True
#python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=False train.window_size=40 exp_name=$exp_name
#python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=False train.window_size=50 exp_name=$exp_name
#python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=False train.window_size=60 exp_name=$exp_name
#python run_worm2vec.py train.n_samples=250000 train.n_epochs=30 test.n_samples=10000 train.batchsize=100 predict.isLoadedimg=False train.window_size=70 exp_name=$exp_name