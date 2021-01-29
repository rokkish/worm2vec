import os
import time
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import json
import pprint
import pandas as pd
import matplotlib.image as mpimg
import torch
import argparse

def test_version():
    """
        3.3.3
        1.16.4
    """
    import matplotlib
    print(matplotlib.__version__)
    print(np.__version__)


def test_plot():
    # config
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca(projection='3d')
    ax.set_box_aspect((1,1,1))
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(-12, 12)

    # sample data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 10)
    x= 10 * np.outer(np.cos(u), np.sin(v))
    y = 10 * np.outer(np.sin(u), np.sin(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
    # print(x.shape, y.shape, z.shape)
    #(100, 10) (100, 10) (100, 10)

    # plot data
    scat1, = ax.plot(x[0,::2],y[0,::2],z[0,::2],alpha=0.5, lw=0, marker="o",color='tab:green')
    scat2, = ax.plot(x[0,1::2],y[0,1::2],z[0,1::2],alpha=0.5, lw=0, marker="o",color='tab:red')

    def init():
        return scat1, scat2, 

    def animate(i):
        scat1.set_data((x[:i,::2].flatten(),y[:i,::2].flatten()))
        scat1.set_3d_properties(z[:i,::2].flatten())
        scat2.set_data((x[:i,1::2].flatten(),y[:i,1::2].flatten()))
        scat2.set_3d_properties(z[:i,1::2].flatten())
        return scat1, scat2, 

    ani = animation.FuncAnimation(fig, animate, 100,
                                    interval=100, init_func=init, blit=True, repeat=True)
    ani.save('./tmp/point_anim.gif', writer="imagemagick",dpi=100)


class Plotter(object):
    def __init__(self, args):
        self.plot_length = args.l
        self.window_name = args.w
        self.data_type = args.d + "data"

        if self.data_type == "testdata":
            date_of_path = "2021-01-27"
            original_path = "alldata_unpublished/alldata"
        elif self.data_type == "traindata":
            date_of_path = "2021-01-29"
            original_path = "alldata_tanimoto/alldata/alldata"
        else:
            raise ValueError("input test, train")

        self.path_to_img = "../outputs/{}/plt_onewindow_3k/plt_{}_win_{:0=2}/projector/after_test/sprite.png".format(date_of_path, args.d, self.window_name)
        self.path_to_original_img_dir = "/root/worm2vec/data/{}".format(original_path)
        self.path_to_metadata = "../outputs/{}/plt_onewindow_3k/plt_{}_win_{:0=2}/projector/after_test/metadata.tsv".format(date_of_path, args.d, self.window_name)
        self.path_to_pcadata = "./tmp/state_{}_plt_onewindow_3k_plt_{}_win_{:0=2}.txt".format(date_of_path, args.d, self.window_name)
        self.save_3dpath = './tmp/gif/point_anim_worm2vec_3d_w{:0=2}_l{:0=4}.gif'.format(self.window_name, self.plot_length)
        self.save_3dpath_mp4 = './tmp/mp4/{}/point_anim_worm2vec_3d_w{:0=2}_l{:0=4}.mp4'.format(self.data_type, self.window_name, self.plot_length)
        self.save_2dpath = './tmp/point_anim_worm2vec_2d.gif'
        self.zoom_rate = 0.8
        self.t_afterimage = 20
        self.single_image_dim = 16
        self.data = self.parse_txt()
        self.file_date, self.metadata_id_list = self.load_metadata()
        self.img = self.load_compressedimg()
        self.img_list = self.load_originalimg()
        self.init_t = time.time()

    def parse_txt(self):
        with open(self.path_to_pcadata) as f:
            json_dict = json.load(f)[0]

        #pprint.pprint(json_dict.keys(), width=40)
        #print(len(json_dict["projections"]))

        df = pd.json_normalize(json_dict["projections"])
        print(df.shape)
        pprint.pprint(df.head())
        
        slice_dimensions = json_dict["pcaComponentDimensions"]
        df = df.iloc[:, slice_dimensions]
        print(df.shape)
        pprint.pprint(df.head())
        return df

    def load_compressedimg(self):
        img = mpimg.imread(self.path_to_img)
        return img

    def load_metadata(self):
        metadata = pd.read_csv(self.path_to_metadata, sep="\t")

        metadata_id_list = metadata.loc[:, "Time"].values

        file_date = set(metadata.loc[:, "Date"])
        assert len(file_date) == 1
        file_date = list(file_date)[0]

        return file_date, metadata_id_list

    def load_originalimg(self):
        img_list = []
        for file_id in self.metadata_id_list:
            file = "{}/{}_{:0=6}.pt".format(self.path_to_original_img_dir, self.file_date, file_id)
            if os.path.isfile(file):
                tmp = torch.load(file)[0, 0].numpy()
                img_list.append(tmp)            
        return img_list

    def spherize(self):
        """Sphereize data. 
            Cite: https://github.com/tensorflow/tensorboard/issues/2421
        """
        columns = self.data.columns
        embedding = self.data.values
        #(3000, 3)

        centroid = np.mean(embedding, axis=0)
        print(centroid.shape)

        for i in range(embedding.shape[0]):
            embedding[i, :] = embedding[i, :] - centroid

        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        self.data = pd.DataFrame(embedding, columns=columns)

    def plot3d(self):
        # sample data
        x = self.data.loc[:, "pca-0"].values
        y = self.data.loc[:, "pca-1"].values
        z = self.data.loc[:, "pca-2"].values
        lim = {
            "x": {
                "min":x.min(),
                "max":x.max()},
            "y": {
                "min":y.min(),
                "max":y.max()},
            "z": {
                "min":z.min(),
                "max":z.max()},
            }
        time = list(map(str, range(self.plot_length)))
        #pprint.pprint(lim)

        # shortcut
        zr = self.zoom_rate
        t_ai = self.t_afterimage
        img_dim = self.single_image_dim

        # config 3d
        fig = plt.figure(figsize=(9, 4))
        ax0 = fig.add_subplot(121, projection="3d")
        ax0.grid(False)
        ax0.set_box_aspect((1, 1, 1))
        ax0.set_xlim(lim["x"]["min"] * zr, lim["x"]["max"] * zr)
        ax0.set_ylim(lim["y"]["min"] * zr, lim["y"]["max"] * zr)
        ax0.set_zlim(lim["z"]["min"] * zr, lim["z"]["max"] * zr)
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        ax0.set_zlabel("z")

        # config img
        imgs = []
        ax1 = fig.add_subplot(122)
        ax1.axis('off')

        # plot all dot
        ax0.scatter(x, y, z, alpha=0.3, color="skyblue", s=1)

        # plot dot for animation
        scat1,  = ax0.plot(x[0], y[0], z[0], alpha=1, lw=0, marker="o", color='black')

        # plot afterimage
        scat2,  = ax0.plot(x[0], y[0], z[0], alpha=0.7, color="black")

        # plot title
        ax0.text2D(0.05, 0.95, "Worm2vec", transform=ax0.transAxes)
        scat3 = ax0.text(0.15, 0.95, 0, time[0], transform=ax0.transAxes, fontsize="medium")

        def init():
            return scat1, scat2, scat3,

        def animate(i):
            scat1.set_data((x[i].flatten(), y[i].flatten()))
            scat1.set_3d_properties(z[i].flatten())

            if i > t_ai:
                scat2.set_data((x[i-t_ai:i].flatten(), y[i-t_ai:i].flatten()))
                scat2.set_3d_properties(z[i-t_ai:i].flatten())
            else:
                scat2.set_data((x[:i].flatten(), y[:i].flatten()))
                scat2.set_3d_properties(z[:i].flatten())

            scat3.set_text(time[i])

            if len(imgs) > 0:
                imgs.pop().remove()

            #N_width = self.img.shape[0]//img_dim
            #h = i//(N_width)
            #w = i-(N_width)*h
            #img = ax1.imshow(self.img[img_dim*h:img_dim*(h+1), img_dim*w:img_dim*(w+1)])
            img = ax1.imshow(self.img_list[i], cmap='gray')
            imgs.append(img)

            print("\r {}/{}".format(i, self.plot_length), end="")

            return scat1, scat2, scat3,

        ani = animation.FuncAnimation(fig, animate, self.plot_length,
                                        interval=50, init_func=init, blit=True, repeat=False)
        ani.save(self.save_3dpath_mp4, writer="ffmpeg",dpi=100)
        #ani.save(self.save_3dpath, writer="imagemagick",dpi=100)

    def plot2d(self):
        # sample data
        x = self.data.loc[:, "pca-0"].values
        y = self.data.loc[:, "pca-1"].values
        z = self.data.loc[:, "pca-2"].values
        lim = {
            "x": {
                "min":x.min(),
                "max":x.max()},
            "y": {
                "min":y.min(),
                "max":y.max()},
            "z": {
                "min":z.min(),
                "max":z.max()},
            }
        #pprint.pprint(lim)

        # config
        fig = plt.figure(figsize=(6,6))
        ax = fig.subplots()
        zr = self.zoom_rate
        #ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(lim["x"]["min"] * zr, lim["x"]["max"] * zr)
        ax.set_ylim(lim["y"]["min"] * zr, lim["y"]["max"] * zr)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # plot data
        scat1, = ax.plot(x[0], y[0], alpha=0.5, lw=0, marker="o", color='tab:green')

        def init():
            return scat1,

        def animate(i):
            scat1.set_data((x[:i].flatten(), y[:i].flatten()))
            print("\r {}/3000".format(i), end="")
            return scat1,

        ani = animation.FuncAnimation(fig, animate, 3000,
                                        interval=10, init_func=init, blit=True, repeat=True)
        ani.save(self.save_2dpath, writer="imagemagick",dpi=100)

def isDebugmode():
    if os.getcwd() == "/root/worm2vec/worm2vec":
        return True
    return False

def main(args):
    if isDebugmode():
        os.chdir("./test/sequential/worm2vec/visualize3d")
    #test_version()
    #test_plot()
    plotter = Plotter(args)
    #plotter.spherize()
    plotter.plot3d()
#    plotter.plot2d()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", type=int, default=100)
    parser.add_argument("-w", type=int, default=20)
    parser.add_argument("-d", type=str, default="train")
    args = parser.parse_args()
    main(args)

