import os
import glob
import math
import itertools
import numpy as np


class Creator(object):
    def __init__(self):
        # 図形ごとの座標
        self.coordinates = {
            "circle": None,
            "square": None,
            "triangle": None,
            }
        # モーフィング図形ごとの座標
        self.morph_coordinates = {
            }

        # 最近傍点のインデックスペア
        self.coordinates_nn = {
            }

        # 図形の点の数
        self.N = 240

        # 全図形の組み合わせ
        self.comb = list(self.get_comb())

        # 補完ステップ数
        self.STEP = 10

        # resize
        self.SIZE = 32

    def create(self):
        # circle, square, triangleの座標を取得
        self.get_coordinates()

        # 最近傍点の探索
        self.search_nearest_neighbors()

        # 最近傍点へのmorphingデータ点の作成
        self.create_morphing_coordinates()

        # 座標にノイズ付与
        self.add_noise()

        # 図形の確認とgifの作成
        self.sample_plt()

        # 読み込むためのpngfileの作成
        self.save_png()

        # png -> ndarray変換
        self.translate_into_np()

    def get_comb(self):
        """need to load at many times
        """
        return itertools.permutations(list(self.coordinates.keys()), r=2)

    def get_coordinates(self):
        """circle, square, triangleの座標を取得
        """
        xy_circle = self.circle(self.N)
        xy_square = self.square(self.N)
        xy_triangle = self.triangle(self.N)

        for i, xy in enumerate([xy_circle, xy_square, xy_triangle]):
            if xy.shape[0] != self.N:
                raise ValueError("num of dot {} is not N, but {}".format(i, xy.shape[0]))

        self.coordinates["circle"] = xy_circle
        self.coordinates["square"] = xy_square
        self.coordinates["triangle"] = xy_triangle

    def circle(self, N):
        xy = []
        radius = 0.25
        c = 0.5
        for theta in np.arange(0, 360, 360/N):
            x = c + radius * math.cos(2*math.pi*theta/360)
            y = c + radius * math.sin(2*math.pi*theta/360)
            xy.append([x, y])
        return np.array(xy)

    def square(self, N):
        line_N = N // 4
        xy = []

        start, end = 0.25, 0.75
        step = abs(end - start) / line_N
        # under
        for dot in np.arange(start, end, step):
            x = dot
            y = start
            xy.append([x, y])
        # right
        for dot in np.arange(start, end, step):
            y = dot
            x = end
            xy.append([x, y])

        start, end = 0.75, 0.25
        step = - abs(end - start) / line_N
        # top
        for dot in np.arange(start, end, step):
            x = dot
            y = start
            xy.append([x, y])
        # left
        for dot in np.arange(start, end, step):
            y = dot
            x = end
            xy.append([x, y])

        return np.array(xy)

    def triangle(self, N):
        line_N = N // 3
        xy = []
        # under
        start, end = 0.25, 0.75
        step = abs(end - start) / line_N
        for dot in np.arange(start, end, step):
            x = dot
            y = start
            xy.append([x, y])
        # right
        start, end = 0.75, 0.5
        step = - abs(end - start) / line_N
        for dot in np.arange(start, end, step):
            x = dot
            y = 1.75 - 2 * dot
            xy.append([x, y])
        # left
        start, end = 0.5, 0.25
        step = -abs(end - start) / line_N
        for dot in np.arange(start, end, step):
            x = dot
            y = - 0.25 + 2 * dot
            xy.append([x, y])

        return np.array(xy)

    def search_nearest_neighbors(self):
        """最近傍点の探索（全図形の組み合わせ）
        """
        for fig_a, fig_b in self.comb:
            coordinates_nn = self.nearest_neighbors_top1(fig_a, fig_b)
            self.coordinates_nn["nn_{}_{}".format(fig_a, fig_b)] = coordinates_nn

    def euclidian_distance(self, a, b):
        """
        Args:
            a (1, 2): ndarray (x, y)
            b (1, 2): ndarray (x, y)

        Returns:
            float: euclidian distance
        """
        ab = np.square(a - b)
        return np.sqrt(ab[0]+ab[1])

    def nearest_neighbors_top1(self, fig_a, fig_b):
        n_xy_a = self.coordinates[fig_a]
        n_xy_b = self.coordinates[fig_b]

        idx_min_distance = []
        for i, xy_a in enumerate(n_xy_a):
            distance_list = []
            for j, xy_b in enumerate(n_xy_b):
                distance_list.append(self.euclidian_distance(xy_a, xy_b))
            min_idx = np.argsort(distance_list)[0]
            idx_min_distance.append([i, min_idx])
        return idx_min_distance

    def create_morphing_coordinates(self):
        """最近傍点へのmorphingデータ点の作成
        """
        for fig_a, fig_b in self.comb:

            xy_comb = []

            for idx_a, idx_b in self.coordinates_nn["nn_{}_{}".format(fig_a, fig_b)]:

                xy_tmp = []
                xy_a = self.coordinates[fig_a][idx_a]
                xy_nn_b = self.coordinates[fig_b][idx_b]

                for step in range(self.STEP + 1):
                    xy_step = self.morphing(xy_a, xy_nn_b, step)
                    xy_tmp.append(xy_step)

                xy_comb.append(xy_tmp)

            self.morph_coordinates["{}_{}".format(fig_a, fig_b)] = np.array(xy_comb)

    def morphing(self, xy_a, xy_b, step):
        return xy_a + (xy_b - xy_a) * step / self.STEP

    def gaussian_noise(self, xy) -> np.array:
        return xy + np.random.normal(0, 0.002, size=xy.shape)

    def add_noise(self):
        for fig_i, fig_j in self.comb:

            xy = self.morph_coordinates["{}_{}".format(fig_i, fig_j)]
            xy = self.gaussian_noise(xy)
            self.morph_coordinates["{}_{}".format(fig_i, fig_j)] = xy

    def sample_plt(self):
        """sample plot
        """
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image

        os.makedirs("./tmp/gif", exist_ok=True)

        for fig_i, fig_j in self.comb:

            xy = self.morph_coordinates["{}_{}".format(fig_i, fig_j)]
            for step in range(self.STEP+1):

                # slice step (self.N, step, xy)
                xy_step = xy[:, step, :]

                patch = patches.Polygon(xy=xy_step, closed=True, fc="black", ec="black")
                shape_name = "test_morphing"

                fig = plt.figure(figsize=(3, 3))
                ax = plt.axes()

                ax.add_patch(patch)

                plt.axis("off")
                ax.set_aspect("equal")
                #plt.title(shape_name)
                #ax.text(0.1, 0.1, "{}_to_{}".format(fig_i, fig_j), size=10)

                plt.savefig("./tmp/gif/{}_{:0=3}.png".format(shape_name, step))

                plt.close()

            # anime to gif
            files = sorted(glob.glob("./tmp/gif/{}_*.png".format(shape_name)))
            images = list(map(lambda file: Image.open(file), files))
            images[0].save("./tmp/gif/out_{}_{}.gif".format(fig_i, fig_j), save_all=True, append_images=images[1:], duration=400, loop=False)

    def save_png(self):
        """save morphing image
        """
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        os.makedirs("./tmp/png", exist_ok=True)

        for fig_i, fig_j in self.comb:

            xy = self.morph_coordinates["{}_{}".format(fig_i, fig_j)]
            for step in range(self.STEP+1):

                # slice step (self.N, step, xy)
                xy_step = xy[:, step, :]

                patch = patches.Polygon(xy=xy_step, closed=True, fc="black", ec="black")

                fig = plt.figure(figsize=(3, 3))
                ax = plt.axes()

                ax.add_patch(patch)

                plt.axis("off")
                ax.set_aspect("equal")

                os.makedirs("./tmp/png/{}_{}".format(fig_i, fig_j), exist_ok=True)
                plt.savefig("./tmp/png/{}_{}/{:0=3}.png".format(fig_i, fig_j, step))

                plt.close()

    def translate_into_np(self):
        from PIL import Image
        os.makedirs("./tmp/np", exist_ok=True)
        for fig_i, fig_j in self.comb:

            folder = "./tmp/png/{}_{}".format(fig_i, fig_j)
            files = sorted(glob.glob("{}/*.png".format(folder)))

            arr = np.zeros((len(files), self.SIZE, self.SIZE))

            for k, file in enumerate(files):
                img = Image.open(file)
                img = img.convert("L")
                img = img.resize((self.SIZE, self.SIZE))
                img = np.array(img)
                arr[k] = img

            np.save("./tmp/np/{}_{}".format(fig_i, fig_j), arr)


def main():
    creator = Creator()
    creator.create()


if __name__ == "__main__":
    main()
