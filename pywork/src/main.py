from cache import sqlite_cache
from utils import output_img
import matplotlib.pyplot as plt
import numpy as np
from const import DIM, N, M, BOLD


def clamp(x, mn, mx):
    return max(mn, min(x, mx))


def get_rotation_matrix(r, theta):  # r, thetaの値から画像を生成
    img = np.zeros((DIM, DIM), dtype=np.float32)
    for i in range(DIM):
        for j in range(DIM):
            x = i - DIM / 2
            y = j - DIM / 2
            dist = abs(x * np.cos(theta) + y * np.sin(theta) - r)
            pix = clamp((1. - dist / BOLD) * 255., 0., 255.)
            img[i, j] = pix
    return img


@sqlite_cache("input.db")
def get_input_images(N, M):
    r_min = 0
    r_max = DIM / 2
    theta_min = 0
    theta_max = np.pi * 2
    data = []
    for i in range(N):
        for j in range(M):
            r = r_min + (r_max - r_min) * i / N
            theta = theta_min + (theta_max - theta_min) * j / M
            img = get_rotation_matrix(r, theta)
            img = img.reshape(DIM * DIM)
            data.append(img)
    return np.array(data).T


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
