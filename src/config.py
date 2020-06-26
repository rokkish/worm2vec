# hypara of model
IMG_SIZE = 64
BATCH_SIZE = 10 #256
tau = 10
# DATA
MAX_LEN_TRAIN_DATA = 16000
NUM_POSITIVE = 3

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
num_of_tensor_to_embed = 300
nrow = 2
