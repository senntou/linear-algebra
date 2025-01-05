from cache import sqlite_cache
from utils import output_img
import matplotlib.pyplot as plt
import numpy as np
from const import DIM, N, M, BOLD
from input import get_input_images


def get_covariance_matrix(data):
    return np.cov(data)


@sqlite_cache("eigen.db")
def get_eigen(cov):
    return np.linalg.eig(cov)


def plot_manifold(y_data):
    x = y_data[0].reshape(N, M)
    y = y_data[1].reshape(N, M)
    z = y_data[2].reshape(N, M)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='bwr')
    plt.savefig("manifold.png")
    plt.close()


if __name__ == '__main__':
    # 画像の読み込み
    print("画像の読み込み")
    data = get_input_images(N, M)  # 各列が画像のベクトル

    # 分散共分散行列を計算
    print("分散共分散行列を計算")
    cov = get_covariance_matrix(data)

    # 固有値と固有ベクトルを計算
    print("固有値と固有ベクトルを計算")
    eigvals, eigvecs = get_eigen(cov)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    # 固有ベクトルを画像に変換して表示
    # print("固有ベクトルを画像に変換して出力")
    # for i in range(10):
    #     output_img(eigvecs[:, i], f"eig{i}.png")

    # 潜在空間への射影
    print("潜在空間への射影")
    DIM_L = 3
    v = eigvecs[:, :DIM_L]
    y = v.T @ data

    # 多様体をプロット
    print("多様体をプロット")
    plot_manifold(y)

    # 観測データを擬似的に生成

    # 観測データを潜在空間に射影

    # r, thetaの値を予測
