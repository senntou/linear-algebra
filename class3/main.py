import os
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    get_x_base,
    get_x_pnt,
    unflatten,
    show_heatmap,
    show_heatmaps,
    plot_3d,
    get_input_image_vector,
)
from utils import DIM, DIR, SPLIT


if __name__ == "__main__":
    # ディレクトリが存在しない場合は作成
    os.makedirs(DIR, exist_ok=True)

    # x_pnt : 1次元のベクトルを作成
    x_base = get_x_base(flatten=True)
    x_pnt = get_x_pnt(x_base)

    # 分散共分散行列
    s = np.zeros((DIM**2, DIM**2))
    x_mean = np.mean(x_pnt, axis=0)
    for x in x_pnt:
        x -= np.mean(x_mean)
        temp = x.reshape(-1, 1) @ x.reshape(1, -1)
        s += temp
    s /= DIM**2

    # 固有値と固有ベクトル
    w, v = np.linalg.eig(s)
    w = np.real(w)
    v = np.real(v)

    # 固有ベクトルを表示
    temp = []
    for i in range(4):
        temp.append(unflatten(v[:, i]))
    show_heatmaps(temp, "eigenvector", save=True)

    ############################################################

    # (2) x_baseを3次元に射影
    X = np.array(x_base).T  # xベクトルが横に並んだ行列
    Y = v[:, :3].T @ X  # 3次元に射影
    plot_3d(Y[0], Y[1], Y[2], "y", save=False)  # 3次元に射影したデータを表示

    ############################################################

    # (3) 観測画像から元画像を復元
    x_input = get_input_image_vector(line=5)
    show_heatmap(unflatten(x_input), "input", save=True)

    # 画像を3次元に射影
    y_input = v[:, :3].T @ x_input

    # 最も近い点を探す
    res_i = 0
    res_lmd = 0
    dist = 1e9
    for i in range(DIM - 1):
        for lmd in np.linspace(0, 1, SPLIT + 1)[:-1]:
            y = Y[:, i] * (1 - lmd) + Y[:, i + 1] * lmd
            d = np.linalg.norm(y - y_input)
            if d < dist:
                dist = d
                res_i = i
                res_lmd = lmd

    # 復元画像を表示
    print(f"res_i: {res_i + 1}, res_lmd: {res_lmd}")
    y_res = v.T @ X[:, res_i] * (1 - res_lmd) + v.T @ X[:, res_i + 1] * res_lmd
    x_res = v @ y_res
    show_heatmap(unflatten(x_res), "reconstruction", save=True)
