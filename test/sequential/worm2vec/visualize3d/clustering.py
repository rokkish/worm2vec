import os
import time
import json
import pprint
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


class Labeler(object):
    def __init__(self, args):
        self.data_type = args.d
        self.window_name = args.w
        self.eps = args.eps
        self.min_samples = args.min_samples
        self.model = args.model
        self.n_clusters = args.n_clusters

        if self.data_type == "test":
            date_of_path = "2021-01-27"
            original_path = "alldata_unpublished/alldata"
        elif self.data_type == "train":
            date_of_path = "2021-01-29"
            original_path = "alldata_tanimoto/alldata/alldata"
        else:
            raise ValueError("input -d test, train")

        self.load_dir = "../outputs/{}/plt_onewindow_3k/plt_{}_win_{:0=2}/projector".format(date_of_path, self.data_type, self.window_name)
        self.path_to_img = "{}/after_test/sprite.png".format(self.load_dir)
        self.path_to_original_img_dir = "/root/worm2vec/data/{}".format(original_path)
        self.path_to_metadata = "{}/after_test/metadata.tsv".format(self.load_dir)
        self.path_to_pcadata = "./tmp/state_{}_plt_onewindow_3k_plt_{}_win_{:0=2}.txt".format(date_of_path, self.data_type, self.window_name)

        if self.model == "dbscan":
            tmp = "{}_eps{}".format(self.model, self.eps)
        elif self.model == "kmeans":
            tmp = "{}_class{}".format(self.model, self.n_clusters)
        else:
            raise ValueError("ERROR --model")
        self.path_to_save_metadata = "./tmp/metadata_clustering/metadata_win{}_{}_{}.csv".format(self.window_name, self.data_type, tmp)

        self.data = self.parse_txt()
        self.file_date, self.metadata_id_list = self.load_metadata()
        self.init_t = time.time()

    def parse_txt(self):
        with open(self.path_to_pcadata) as f:
            json_dict = json.load(f)[0]

        df = pd.json_normalize(json_dict["projections"])
        slice_dimensions = json_dict["pcaComponentDimensions"]
        df = df.iloc[:, slice_dimensions]
        print(df.shape)
        pprint.pprint(df.head())
        return df

    def load_metadata(self):
        metadata = pd.read_csv(self.path_to_metadata, sep="\t")

        metadata_id_list = metadata.loc[:, "Time"].values

        file_date = set(metadata.loc[:, "Date"])
        assert len(file_date) == 1
        file_date = list(file_date)[0]

        return file_date, metadata_id_list

    def labelling(self):
        arr = self.data.values
        print(arr.shape)
        if self.model == "dbscan":
            self.clustering_label = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(arr).labels_
        elif self.model == "kmeans":
            self.clustering_label = KMeans(n_clusters=self.n_clusters, random_state=0).fit(arr).labels_

    def save_metadata(self):
        dic = {
            self.model: self.clustering_label
        }
        df = pd.DataFrame(dic)
        df.to_csv(self.path_to_save_metadata)

def add_label(save_dir, result_dir, path_to_metadata, path_to_clustering_labels):

    save_path = save_dir + result_dir + "/metadata.tsv"
    path_to_8vector_labels = "./tmp/metadata_8vector/metadata_win20_j2.0_test.csv"

    metadata = pd.read_csv(path_to_metadata, sep="\t")
    label = pd.read_csv(path_to_8vector_labels, index_col=0)
    label2 = pd.read_csv(path_to_clustering_labels, index_col=0)

    metadata = metadata.reset_index(drop=True)
    label = label.reset_index(drop=True)
    label2 = label2.reset_index(drop=True)

    df = pd.concat([metadata, label, label2], axis=1)
    print(df.head())

    df.to_csv(save_path, header=True, index=False, sep="\t")


def main(args):
    labeler = Labeler(args)
    labeler.labelling()
    labeler.save_metadata()

    import shutil
    save_dir = labeler.load_dir
    result_dir = "/add_" + labeler.path_to_save_metadata.split("/")[-1].split(".csv")[0]
    shutil.copytree(save_dir + "/after_test", save_dir + result_dir)

    add_label(save_dir, result_dir, labeler.path_to_metadata, labeler.path_to_save_metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", type=int, default=20)
    parser.add_argument("-d", type=str, default="train")
    parser.add_argument("-j", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--model", type=str, default="dbscan")
    parser.add_argument("--n_clusters", type=int, default=5)
    args = parser.parse_args()
    main(args)
