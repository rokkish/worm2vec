"""Predictor class mainly predict without training data.
"""
import os
import time
import wandb
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import get_logger
logger = get_logger.get_logger(name="predictor")
from visualize.sprite_images import create_sprite_image, save_sprite_image
from tensorboard.plugins import projector

class Predictor(object):
    """class for trained sequential model.
    """
    def __init__(self,
                 params,
                 loss,
                 optim,
                 train_op,
                 placeholders):

        # initialize
        self.loss = loss
        self.optim = optim
        self.train_op = train_op

        # yaml config
        self.lr = params.train.learning_rate
        self.n_epochs = params.train.n_epochs
        self.batchsize = params.train.batchsize
        self.dim = params.train.dim
        self.test_original_data_path = params.dir.test.original
        self.checkpoint = params.dir.checkpoint
        self.checkpoint_fullpath = "/root/worm2vec/worm2vec/test/sequential/worm2vec/outputs/{}/checkpoints/model.ckpt".format(params.dir.checkpoint_datetime)
        self.default_logdir = params.dir.tensorboard
        self.size = params.predict.img_size
        self.output_dim = params.predict.dim_out
        self.n_embedding = params.predict.n_embedding
        self.isLoadedimg = params.predict.isLoadedimg

        # placeholders
        self.x_previous = placeholders["x_previous"]
        self.x_now = placeholders["x_now"]
        self.x_next = placeholders["x_next"]
        self.learning_rate = placeholders["learning_rate"]

        # summary_writer
        self.train_summary_writer = tf.summary.FileWriter("./tensorboard/train")
        self.test_summary_writer = tf.summary.FileWriter("./tensorboard/test")

        # Configure tensorflow session
        self.init_global = tf.global_variables_initializer()
        self.init_local = tf.local_variables_initializer()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = params.device.gpu.allow_growth
        self.config.gpu_options.visible_device_list = params.device.gpu.id
        self.config.log_device_placement = params.device.gpu.log_device_placement

        ### predict ###
        self.constant_idx = 0
        self.skip_count = 0
        self.single_view_mode = params.predict.single_view_mode
        self.layers_name = [
            "concat_inputs/concat:0",
            "context_encoder/MatMul:0",
            "concat_outputs/concat:0"
        ]

        self.tensor_path_name = "tensor.csv"
        self.metadata_path_name = "metadata.tsv"
        self.sprite_image_path_name = "sprite.png"
        self.embedding_column_name = "ShapePrev_ShapeNext_PrevContShapeNow_NextContShapeNow_ContShapeNow"


    def restore_session(self):
        """Restore tensorflow Session

        Returns:
            sess: load inital config
        """
        sess = tf.Session(config=self.config)
        sess.run([self.init_global,
                  self.init_local],
                 feed_dict={})

        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoint_fullpath)

        return saver, sess

    def fit(self, data):

        saver, sess = self.restore_session()

        batcher = self.minibatcher_withlabel(data["test_x"], data["test_label"])

        self.logdir = "{}projector/after_test/".format(self.default_logdir)
        os.makedirs(self.logdir, exist_ok=True)

        medium_op = list(
            map(
                lambda tname: tf.get_default_graph().get_tensor_by_name(tname),
                self.layers_name)
            )
        # def zero vec
        prev_context_summary_np = np.zeros((self.n_embedding, self.output_dim))
        next_context_summary_np = np.zeros((self.n_embedding, self.output_dim))
        prev_target_summary_np = np.zeros((self.n_embedding, self.output_dim))
        next_target_summary_np = np.zeros((self.n_embedding, self.output_dim))
        conttarget_summary_np = np.zeros((self.n_embedding, self.output_dim))
        prev_context_img = np.zeros((self.n_embedding, self.size, self.size))
        next_context_img = np.zeros((self.n_embedding, self.size, self.size))
        target_img = np.zeros((self.n_embedding, self.size, self.size))
        conttarget_img = np.zeros((self.n_embedding, self.size, self.size))

        labels = {
            "id": [],
            "date": []
        }
        bs = self.batchsize

        for batch, (x_previous, x_now, x_next, x_label, x_date) in enumerate(batcher):

            if batch*bs == self.n_embedding:
                break

            feed_dict = {
                self.x_previous: x_previous,
                self.x_now: x_now,
                self.x_next: x_next,
                self.learning_rate: self.lr
            }

            embedding_summary = sess.run([
                medium_op
                ], feed_dict=feed_dict)

            ## project embedding ##

            # select last layer output
            summary_np = np.array(embedding_summary[0][-1])
            # select cont layer output
            summary_np3 = np.array(embedding_summary[0][1])

            # cat several vectors (output of model)
            prev_context_summary_np[batch*bs: (batch+1)*bs] = summary_np[self.constant_idx: self.constant_idx + 1*bs]
            next_context_summary_np[batch*bs: (batch+1)*bs] = summary_np[self.constant_idx + 1*bs: self.constant_idx + 2*bs]
            prev_target_summary_np[batch*bs: (batch+1)*bs] = summary_np[self.constant_idx + 2*bs: self.constant_idx + 3*bs]
            next_target_summary_np[batch*bs: (batch+1)*bs] = summary_np[self.constant_idx + 3*bs: self.constant_idx + 4*bs]
            conttarget_summary_np[batch*bs: (batch+1)*bs] = summary_np3[self.constant_idx: self.constant_idx + 1*bs]

            # cat several images
            if self.isLoadedimg:
                for idx, x_label_i in enumerate(x_label):
                    prev_img, next_img, now_img = self.load_original_img(x_label_i, x_date)
                    prev_context_img[batch*bs + idx] = prev_img
                    next_context_img[batch*bs + idx] = next_img
                    target_img[batch*bs + idx] = now_img
                    conttarget_img[batch*bs + idx] = now_img

            labels["id"].extend(x_label)
            labels["date"].append(x_date)

        if self.single_view_mode:
            cat_tensors = conttarget_summary_np
            cat_images = conttarget_img
            cat_labels = {
                "id": labels["id"],
                "date": labels["date"]
            }
        else:
            cat_tensors = np.concatenate([
                prev_context_summary_np,
                next_context_summary_np,
                prev_target_summary_np,
                next_target_summary_np,
                conttarget_summary_np
                ], axis=0)
            cat_images = np.concatenate([
                prev_context_img,
                next_context_img,
                target_img,
                target_img,
                conttarget_img
                ], axis=0)  
            cat_labels = {
                "id": labels["id"] + labels["id"] + labels["id"] + labels["id"] + labels["id"],
                "date": labels["date"] + labels["date"] + labels["date"] + labels["date"] + labels["date"]
            }

        variables = tf.Variable(cat_tensors, trainable=False, name="embedding")

        self.create_tensor_tsv(cat_tensors, path="{}{}".format(self.logdir, self.tensor_path_name))

        self.create_metadata_csv(cat_labels, path="{}{}".format(self.logdir, self.metadata_path_name))

        self.mk_spriteimage(cat_images, path="{}{}".format(self.logdir, self.sprite_image_path_name))

        config_projector = self.set_config_projector(variables)
        summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
        projector.visualize_embeddings(summary_writer, config_projector)

    def minibatcher_withlabel(self, inputs, labels):
        """

        Args:
            inputs (list): list of data path. (*.npy)
            labels (dataframe): 
                columns: 'Label_date', 'Label_id', 'Anchor_Pos_Neg'

        Yields:
            x_previous, x_now, x_next, t_now_lisst, date (ndarray):
                ([dim]), ([dim]), ([dim]), ([int]), (int)
        """

        dim = self.dim
        bs = self.batchsize
        dates = set(labels["Label_date"])
        window_size = 1

        for i, date in enumerate(dates):

            label_where_date = labels[labels["Label_date"]==date]
            idx_begin = label_where_date.index[0]
            idx_end = label_where_date.index[-1]

            # prev, nextを参照するため. from w to N-w
            indices = np.arange(idx_begin + window_size, idx_end - window_size)
            np.random.shuffle(indices)

            stock_x = 0
            x_prev = np.zeros((bs, dim))
            x_now = np.zeros((bs, dim))
            x_next = np.zeros((bs, dim))
            t_now_list = []

            for jdx in indices:

                if stock_x == 0:
                    x_prev = np.zeros((bs, dim))
                    x_now = np.zeros((bs, dim))
                    x_next = np.zeros((bs, dim))
                    t_now_list = []
                    date_list = []

                jdx_prev, jdx_now, jdx_next = jdx - 1, jdx, jdx + 1
                t_prev = labels.iloc[jdx_prev, 1]
                t_now = labels.iloc[jdx_now, 1]
                t_next = labels.iloc[jdx_next, 1]

                if self.t_is_continuous(t_prev, t_now, t_next, window_size):

                    tmp_prev = np.reshape(inputs[jdx_prev], (1, dim))
                    tmp_now = np.reshape(inputs[jdx_now], (1, dim))
                    tmp_next = np.reshape(inputs[jdx_next], (1, dim))

                    x_prev[stock_x] = tmp_prev
                    x_now[stock_x] = tmp_now
                    x_next[stock_x] = tmp_next
                    t_now_list.append(t_now)

                    stock_x += 1

                else:
                    continue

                if stock_x == bs:
                    stock_x = 0
                    yield x_prev, x_now, x_next, t_now_list, date

    def t_is_continuous(self, t_prev, t_now, t_next, window_size):
        if t_now - t_prev == window_size and t_next - t_now == window_size:
            return True
        self.skip_count += 1
        return False

    @staticmethod
    def load_pt_to_np(path):
        return torch.load(path).numpy()

    def reshape_np(self, arr):
        arr = np.reshape(arr[0, 0], (1, self.size, self.size))
        return arr

    def get_original_data_path(self, date, time):
        return "{}/{}_{:0=6}.pt".format(self.test_original_data_path, date, time)

    def load_original_img(self, x_time, x_date):
        """
        Returns:
           prev, next, now (ndarray): (1, dim, dim)
        """
        # t_is_continuousより下記で参照してもOK
        prev_t, now_t, next_t = x_time - 1, x_time, x_time + 1
 
        prev_path = self.get_original_data_path(x_date, prev_t)
        next_path = self.get_original_data_path(x_date, next_t)
        now_path = self.get_original_data_path(x_date, now_t)

        prev_img = self.reshape_np(self.load_pt_to_np(prev_path))
        next_img = self.reshape_np(self.load_pt_to_np(next_path))
        now_img = self.reshape_np(self.load_pt_to_np(now_path))

        return prev_img, next_img, now_img

    def create_tensor_tsv(self, cat_tensors, path):
        # make tensor.tsv
        df = pd.DataFrame(cat_tensors).astype("float64")
        df.to_csv(path, header=False, index=False, sep="\t")

    def create_metadata_csv(self, cat_labels, path):
        # make metadata.tsv (labels)
        with open(path, "w") as f:
            f.write("Time\tDate\t{}\n".format(self.embedding_column_name))

            for index in range(len(cat_labels["id"])):
                id_ = cat_labels["id"][index]
                multi_label = index // self.n_embedding
                date_ = cat_labels["date"][multi_label]
                f.write("%d\t%d\t%d\n" % (id_, date_, multi_label))

    def mk_spriteimage(self, cat_images, path):
        # make sprite image (labels)
        cat_images = (1 - cat_images) * 255.
        save_sprite_image(create_sprite_image(cat_images), path=path)

    def set_config_projector(self, variables):
        # config of projector
        config_projector = projector.ProjectorConfig()
        embedding = config_projector.embeddings.add()
        embedding.tensor_name = variables.name

        # tsv path
        embedding.tensor_path = self.tensor_path_name
        embedding.metadata_path = self.metadata_path_name

        # config of sprite image
        embedding.sprite.image_path = self.sprite_image_path_name
        embedding.sprite.single_image_dim.extend([self.size, self.size])
        return config_projector
