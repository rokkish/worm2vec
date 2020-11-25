# hypara of model
IMG_SIZE = 64
BATCH_SIZE = 1 #256
tau = 10
# hypara of loss
MARGIN_TRIPLET = 10.0
# DATA
MAX_LEN_TRAIN_DATA = 60000
NUM_POSITIVE = 35
NUM_NEGATIVE = 5

# set default of args
epoch = 15
logdir = "default"
gpu_id = "0"
train_dir = "processed/alldata"
model_name = "cbow_test_model"
model = "cbow"
loss_function_name = "binarycrossentropyLoss"
max_predict = 3
max_analyze = 2
zsize = 32

# config of predict.py
nrow = 20

# config of get_distance_table.py
error_idx = -1
