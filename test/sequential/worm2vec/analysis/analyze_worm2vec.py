"""
    0. Spherize data.
    1. Caliculate distance and Clustering Spherized data top-3.
    2. Plot transition probabilities at heatmap.
"""
from io import BytesIO
import os
import glob
import json
from os.path import basename
import hydra
import pprint
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from sklearn.cluster import KMeans


class Analyzer(object):
    def __init__(self, cfg):
        self.path_to_load_vector = cfg.path.load_vector
        self.path_to_metadata = "projector/after_test/metadata.tsv"
        self.path_to_tensor = "projector/after_test/tensor.csv"
        self.path_to_save_data = cfg.path.save_data
        self.window = cfg.calc.window
        self.topk = cfg.calc.topk
        self.model = cfg.clustering.model
        self.k = cfg.clustering.k
        self.thres = cfg.plot.thres
        self.lower_bound = cfg.plot.lower_bound
        self.heat_annot = cfg.plot.heat_annot
        self.isNormed = cfg.plot.norm
        self.figsize = cfg.plot.figsize
        self.digraph_lower_bound = cfg.plot.digraph.lower_bound
        self.norm = "unnorm"
        if self.isNormed:
            self.norm = "norm"

        self.predict_dirs = self.glob_predict_dir()
        print(f"CWD: {os.getcwd()}")
        print(f"GLOBS: {self.predict_dirs}")

        self.metadata = {}
        self.tensor = {}
        self.spherized_tensor = {}
        self.time_date_dist = {}
        self.time_date_dist_class = {}

    def glob_predict_dir(self):
        return glob.glob(f"{self.path_to_load_vector}/*")

    def load_vector(self):
        for path_to_predict in self.predict_dirs:
            glob_name = path_to_predict.split("/")[-1]
            print(f"GLOB: {glob_name}")
            self.metadata[glob_name] = pd.read_csv(f"{path_to_predict}/{self.path_to_metadata}", sep="\t")
            self.tensor[glob_name] = pd.read_csv(f"{path_to_predict}/{self.path_to_tensor}", header=None, sep="\t")

    def slice_topk(self, df):
        return df.iloc[:, :self.topk]

    def spherize(self):
        """Sphereize data. 
            Cite: https://github.com/tensorflow/tensorboard/issues/2421
        """
        for glob_name, df in self.tensor.items():
            
            df = self.slice_topk(df)

            columns = df.columns
            embedding = df.values

            centroid = np.mean(embedding, axis=0)

            for i in range(embedding.shape[0]):
                embedding[i, :] = embedding[i, :] - centroid

            embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)

            spherized = pd.DataFrame(embedding, columns=columns)

            self.spherized_tensor[glob_name] = spherized

    def test_spherize(self):
        self.spherize()

        for k, v in self.spherized_tensor.items():
            print(f"SPHERIZED: {k}")
            print(v.head())

    def vector_between_points(self, df):
        df_now = df.iloc[:-self.window, :].reset_index(drop=True)
        df_next = df.iloc[self.window:, :].reset_index(drop=True)
        return df_next - df_now

    def euclid_distance(self, df):
        """
            Args:
                pd.DataFrame: columns = (x, y, z)
            Return:
                pd.Series: euclid distance.
        """
        df_sub = self.vector_between_points(df)
        df_dist = df_sub.pow(2).sum(axis=1).pow(0.5)
        df_dist.columns = ["dist"]
        return df_dist
    
    def append_zero_last(self, df_dist_in_a_date):
        """
            Args:
                pd.Series: euclid distance. N
            Return:
                pd.Series: euclid distance. N+1
        """
        dist_in_a_date = df_dist_in_a_date.values
        col = df_dist_in_a_date.columns
        arr = np.append(dist_in_a_date, [-1]*self.window)
        return pd.DataFrame(arr, columns=col)

    def calc_distance(self):
        for (glob_name, tensor), (_, metadata) in zip(self.spherized_tensor.items(), self.metadata.items()):
            print(f"CALC DIST: {glob_name}")
            data = pd.concat([metadata.reset_index(drop=True), tensor.reset_index(drop=True)], axis=1)
            dates = set(metadata.loc[:, "Date"].values)
            print(f"NUM OF DATE: {len(dates)}")

            df_in_a_date = pd.DataFrame(None)
            df_time = pd.DataFrame(None)
            df_date = pd.DataFrame(None)
            df_dist = pd.DataFrame(None)
            for date in dates:
                df_time_in_a_date = data[data["Date"]==date].loc[:, "Time"]
                df_date_in_a_date = pd.DataFrame([date]*df_time_in_a_date.shape[0], columns=["Date"])

                df_xyz = data[data["Date"]==date].iloc[:, -self.topk:]
                df_dist_in_a_date = self.euclid_distance(df_xyz)
                df_dist_in_a_date = self.append_zero_last(df_dist_in_a_date)

                df_time = pd.concat([df_time, df_time_in_a_date], axis=0).reset_index(drop=True)
                df_date = pd.concat([df_date, df_date_in_a_date], axis=0).reset_index(drop=True)
                df_dist = pd.concat([df_dist, df_dist_in_a_date], axis=0).reset_index(drop=True)

            self.time_date_dist[glob_name] = pd.concat([df_time, df_date, df_dist], axis=1)
            self.time_date_dist[glob_name].columns = ["Time", "Date", "Dist"]

    def test_calc_distance(self):
        self.calc_distance()

        for k, v in self.time_date_dist.items():
            print(f"SPHERIZED: {k}")
            print(v.shape)
            print(v.head())

    def labelling(self, arr):
        if self.model == "kmeans":
            return KMeans(n_clusters=self.k, random_state=0).fit(arr).labels_
        else:
            raise ValueError("invalid labelling model")

    def plt_inertia_for_optimized_num_of_cluster(self):

        for glob_name, df in self.tensor.items():

            print(f"Inertia: {glob_name}")

            arr = df.values

            if self.model == "kmeans":
                inertias = []
                for i in range(1, self.k+1):
                    km = KMeans(n_clusters=i, random_state=0).fit(arr)
                    inertias.append(km.inertia_)

                plt.plot(range(1, self.k+1), inertias, marker='o')
                plt.title("inertia")
                plt.xlim(1, self.k)
                plt.xlabel("Number of clusters")
                plt.ylabel("Distortion")
                plt.savefig(f"./inertia_{glob_name}.pdf")
                plt.close()

            else:
                raise ValueError("invalid labelling model")

    def plt_silhouette_for_optimized_num_of_clusters(self):
        from sklearn.metrics import silhouette_samples
        from matplotlib import cm

        for glob_name, df in self.tensor.items():

            print(f"SILHOUETTE: {glob_name}")

            arr = df.values

            if self.model == "kmeans":
                for k in range(2, self.k+1):
                    km = KMeans(n_clusters=k, random_state=0).fit(arr).labels_

                    ### silhouette_analysis
                    cluster_labels = np.unique(km)
                    n_clusters = cluster_labels.shape[0]

                    silhouette_vals = silhouette_samples(arr, km, metric='euclidean')
                    y_ax_lower, y_ax_upper= 0, 0
                    yticks = []

                    for i, c in enumerate(cluster_labels):
                            c_silhouette_vals = silhouette_vals[km==c]
                            c_silhouette_vals.sort()
                            y_ax_upper += len(c_silhouette_vals)
                            color = cm.jet(float(i)/n_clusters)
                            plt.barh(range(y_ax_lower,y_ax_upper),            # 水平の棒グラフのを描画（底辺の範囲を指定）
                                            c_silhouette_vals,               # 棒の幅（1サンプルを表す）
                                            height=1.0,                      # 棒の高さ
                                            edgecolor='none',                # 棒の端の色
                                            color=color)                     # 棒の色
                            yticks.append((y_ax_lower+y_ax_upper)/2)          # クラスタラベルの表示位置を追加
                            y_ax_lower += len(c_silhouette_vals)              # 底辺の値に棒の幅を追加

                    silhouette_avg = np.mean(silhouette_vals)                 # シルエット係数の平均値
                    plt.axvline(silhouette_avg,color="red",linestyle="--")    # 係数の平均値に破線を引く 
                    plt.yticks(yticks, cluster_labels + 1)                     # クラスタレベルを表示
                    plt.ylabel('Cluster')
                    plt.xlabel('silhouette coefficient')
                    plt.savefig(f"./silhouette_{k}_{glob_name}.pdf")
                    plt.close()

            else:
                raise ValueError("invalid labelling model")

    def clustering(self):
        for glob_name, df in self.time_date_dist.items():
            print(f"Label: {glob_name}")

            clustering_label = self.labelling(self.tensor[glob_name].values)
            df_class = pd.DataFrame({self.model: clustering_label})
            self.time_date_dist_class[glob_name] = pd.concat([df, df_class], axis=1)
            self.time_date_dist_class[glob_name].columns = ["Time", "Date", "Dist", "Class"]

    def test_clustering(self):
        self.clustering()
        for k, v in self.time_date_dist_class.items():
            print(f"CLUSTERING: {k}")
            print(v.shape)
            print(v.head())

    def cp_original_tensorboard_logs(self, glob_name):
        import shutil
        shutil.copytree(f"{self.path_to_load_vector}/{glob_name}/projector/after_test", f"{self.path_to_save_data}/{glob_name}")

    def save_for_tensorboard(self):
        """Copy and Overwrite metadata.tsv
        """
        for glob_name, df in self.time_date_dist_class.items():
            self.cp_original_tensorboard_logs(glob_name)
            save_path = f"{self.path_to_save_data}/{glob_name}/metadata.tsv"
            df.to_csv(save_path, header=True, index=False, sep="\t")

    def sift_for_trans_prob(self, df):
        return np.append(df.iloc[self.window:].loc[:, "Class"].values, [-1]*self.window)

    def trans_prob(self, df):
        class_next = self.sift_for_trans_prob(df)

        df_tmp = pd.DataFrame(class_next, columns=["class_next"])
        df_transprob = pd.concat([df, df_tmp], axis=1)
        df_transprob = df_transprob.loc[:, ["Class", "class_next", "Dist"]]
        df_transprob.columns = ["now", "next", "Dist"]

        df_transprob  = df_transprob[df_transprob.loc[:, "Dist"] < self.thres]
        df_transprob  = df_transprob[df_transprob.loc[:, "Dist"] > self.lower_bound]

        dict_trans_prob = dict(df_transprob.iloc[:, :-1].value_counts())

        arr_trans_prob = np.zeros((max(max(dict_trans_prob))+1, max(max(dict_trans_prob))+1))
        for (now_, next_), val in dict_trans_prob.items():
            arr_trans_prob[int(now_), int(next_)] = val

        if self.isNormed:
            for i in range(arr_trans_prob.shape[0]):
                arr_trans_prob[i, :] = arr_trans_prob[i, :] / sum(arr_trans_prob[i, :])

        return arr_trans_prob

    def trans_prob_reverse(self, df):
        class_next = self.sift_for_trans_prob(df)

        df_tmp = pd.DataFrame(class_next, columns=["class_next"])
        df_transprob = pd.concat([df, df_tmp], axis=1)
        df_transprob = df_transprob.loc[:, ["Class", "class_next", "Dist"]]
        df_transprob.columns = ["now", "next", "Dist"]

        df_transprob  = df_transprob[df_transprob.loc[:, "Dist"] < self.thres]
        df_transprob  = df_transprob[df_transprob.loc[:, "Dist"] > self.lower_bound]

        dict_trans_prob = dict(df_transprob.iloc[:, :-1].value_counts())

        arr_trans_prob_rev = np.zeros((max(max(dict_trans_prob))+1, max(max(dict_trans_prob))+1))
        for (now_, next_), val in dict_trans_prob.items():
            arr_trans_prob_rev[int(now_), int(next_)] = val

        if self.isNormed:
            for j in range(arr_trans_prob_rev.shape[1]):
                arr_trans_prob_rev[:, j] = arr_trans_prob_rev[:, j] / sum(arr_trans_prob_rev[:, j])

        return arr_trans_prob_rev

    def plot_heatmap(self):
        for glob_name, df in self.time_date_dist_class.items():
            arr_trans_prob = self.trans_prob(df)
            arr_trans_prob_rev = self.trans_prob_reverse(df)

            fig = plt.figure(figsize=(2*self.figsize, self.figsize))

            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_title("trans_prob")
            ax1.set_xlabel("next")
            ax1.set_ylabel("now")
            sns.heatmap(arr_trans_prob, ax=ax1, cmap="GnBu", annot=self.heat_annot, square=True)

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_title("reverse")
            ax2.set_xlabel("next")
            ax2.set_ylabel("now")
            sns.heatmap(arr_trans_prob_rev, ax=ax2, cmap="GnBu", annot=self.heat_annot, square=True)

            #, fmt="g")
            plt.savefig(f"./matrix_thres_from_{self.lower_bound}_to_{self.thres}_{self.norm}_{glob_name}.pdf")
            plt.close()

    def plot_digraph(self):
        from graphviz import Graph, Digraph

        for glob_name, df in self.time_date_dist_class.items():
            arr_trans_prob = self.trans_prob(df)

            dg = Digraph(format="pdf")

            # edge
            for now_ in range(arr_trans_prob.shape[0]):
                for next_ in range(arr_trans_prob.shape[1]):
                    value = arr_trans_prob[now_][next_]
                    if now_ == arr_trans_prob.shape[0] or next_ == arr_trans_prob.shape[1]:
                        pass
                    elif value <= self.digraph_lower_bound:
                        pass
                    elif value == None:
                        pass
                    #elif now_ == next_:
                    #    pass
                    else:
                        dg.edge(str(now_), str(next_), label="{:0=.2f}".format(value))

            dg.render(f"./digraph_{glob_name}")

    def plot_euclid_distance_hist(self):
        for glob_name, df in self.time_date_dist.items():
            df_dist = df.loc[:, "Dist"]
            df_dist.hist(bins=100, weights=np.ones(len(df_dist))/len(df_dist), cumulative=True, density=True)
            plt.xlabel("euclid distance")
            plt.ylabel("histogram")
            plt.xlim(0., 2.)
            #plt.yticks(np.arange(0, 1.01, step=0.1))
            plt.savefig(f"./hist_dist_{glob_name}.pdf")
            plt.close()

    def create_csv_for_gephi(self, glob_name):
        gephi_df = pd.DataFrame(None)

        metadata_path = f"{self.path_to_save_data}/{glob_name}/metadata.tsv"
        metadata_df = pd.read_csv(metadata_path, sep="\t")

        image_path_df = metadata_df.loc[:, "Time"].astype("int").astype("str")
        tmp_path = "./imgs/"
        tmp_path2 = ".png"
        image_path_df = image_path_df + tmp_path2

        sauce = metadata_df.loc[:, "Class"]
        target = pd.DataFrame(self.sift_for_trans_prob(metadata_df))

        gephi_df = pd.concat([
            gephi_df, 
            metadata_df.loc[:, ["Date", "Dist"]],
            sauce,
            target,
            image_path_df
            ], axis=1)

        gephi_df = gephi_df[gephi_df["Dist"]!=-1]

        gephi_path = f"{self.path_to_save_data}/{glob_name}/data_for_gephi.csv"
        gephi_df.columns = ["Date", "Dist", "Source", "Target", "image"]
        gephi_df.to_csv(gephi_path, index=False)
        return gephi_df

    def parse_pbtxt(self, path):
        #TODO: autocheck
        dim = 16
        return dim

    def split_img_and_save(self, glob_name, gephi_df):
        sprite_path = f"{self.path_to_save_data}/{glob_name}/sprite.png"
        pbtxt_path = f"{self.path_to_save_data}/{glob_name}/projector_config.pbtxt"
        dim = self.parse_pbtxt(pbtxt_path)

        with open(sprite_path, "rb") as f:
            binary = f.read()
        img = Image.open(BytesIO(binary))
        img_arr = np.asarray(img)

        path_to_save_split_img = f"{self.path_to_save_data}/{glob_name}/imgs"
        os.makedirs(path_to_save_split_img, exist_ok=True)

        H, W = img_arr.shape[0]//dim, img_arr.shape[1]//dim

        for i in range(gephi_df.shape[0]):
            basename = os.path.basename(gephi_df.loc[:, "image"].iloc[i])
            path_to_save_img_i = f"{path_to_save_split_img}/{basename}"
            wi = (i % W) * dim
            hi = (i // H) * dim
            arr_i = img_arr[hi: hi + dim, wi: wi + dim]
            img_i = Image.fromarray(arr_i)
            img_i.save(path_to_save_img_i, format="png")

    def process_for_gephi(self):
        """
            1.1 transfer .csv from metadat.tsv
            1.2 concat csv, image path list
            2. split sprite.png into each img_id.png
        """
        for glob_name, df in self.time_date_dist_class.items():
            
            gephi_df = self.create_csv_for_gephi(glob_name)
            self.split_img_and_save(glob_name, gephi_df)

@hydra.main(config_name="config")
def main(cfg: DictConfig):
    analyzer = Analyzer(cfg)
    analyzer.load_vector()
    analyzer.spherize()
    analyzer.calc_distance()
#    analyzer.plt_inertia_for_optimized_num_of_cluster()
#    analyzer.plt_silhouette_for_optimized_num_of_clusters()
    analyzer.clustering()
    analyzer.save_for_tensorboard()
    analyzer.process_for_gephi()
    analyzer.plot_euclid_distance_hist()
    analyzer.plot_heatmap()
    analyzer.plot_digraph()

if __name__ == "__main__":
    main()