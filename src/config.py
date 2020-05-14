# hypara of model
IMG_SIZE = 64
layer_count = 4
BATCH_SIZE = 1 #256
# DATA
MAX_LEN_TRAIN_DATA = 3000
MAX_LEN_EVA_LDATA = 10
# 例外処理
error_idx = -1

# set default of args
epoch = 15
logdir = "default"
gpu_id = "0"
train_dir = "processed/alldata"
window = 3
model_name = "cbow_test_model"
model = "cbow"
loss_function_name = "binarycrossentropyLoss"
max_predict = 3
zsize = 32
# config of predict.py
num_of_tensor_to_embed = 300
nrow = 10