# all setting
test_n=120_000
predict_n=100_000

### traindata plot
datatype="train"
test_data="/root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant/"$datatype"/tensor.csv"
test_metadata="/root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant/$datatype/metadata.tsv"
test_original="/root/worm2vec/data/alldata_tanimoto/alldata/alldata"

ws=1
datetime="2021-02-07/fulltrain_stimulus_25k_w20"
exp_name="plt_100k_"$datatype"_win_20_stimulus"

python run_worm2vec.py train.window_size=$ws dir.checkpoint_datetime=$datetime exp_name=$exp_name train.n_samples=100 test.n_samples=$test_n predict.n_embedding=$predict_n train.batchsize=100 predict.isLoadedimg=True train.train_mode=false predict.single_view_mode=True dir.test.data=$test_data dir.test.metadata=$test_metadata dir.test.original=$test_original


## testdata plot
datatype="test"
test_data="/root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant/"$datatype"/tensor.csv"
test_metadata="/root/worm2vec/data/variety_data_strict_r36_n100_rotation_invariant/$datatype/metadata.tsv"
test_original="/root/worm2vec/data/alldata_unpublished/alldata"

ws=1
datetime="2021-02-07/fulltrain_nostimulus_19k_w20"
exp_name="plt_100k_"$datatype"_win_20_nostimulus"

python run_worm2vec.py train.window_size=$ws dir.checkpoint_datetime=$datetime exp_name=$exp_name train.n_samples=100 test.n_samples=$test_n predict.n_embedding=$predict_n train.batchsize=100 predict.isLoadedimg=True train.train_mode=false predict.single_view_mode=True dir.test.data=$test_data dir.test.metadata=$test_metadata dir.test.original=$test_original
