from line_prediction_mini import predict_model
from utils.utils import output_img
import matplotlib.pyplot as plt
import numpy as np
from utils.const import DIM, DIM_L, N, M, BOLD
from utils.input import get_input_images, get_rotation_matrix
from utils.calc import get_covariance_matrix, get_eigen, get_eigvals_of_lines


def plot_manifold(y_data):
    x = y_data[0].reshape(N, M)
    y = y_data[1].reshape(N, M)
    z = y_data[2].reshape(N, M)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='bwr')
    plt.savefig("output/manifold.png")
    plt.close()


if __name__ == '__main__':
    print("\n=== main.py ===")

    # 画像の読み込み
    print("画像の読み込み")
    data = get_input_images(N, M)  # 各列が画像のベクトル

    # 固有値と固有ベクトルを計算
    print("固有値と固有ベクトルを計算")
    eigvals, eigvecs = get_eigvals_of_lines(N, M)

    # 固有値を出力
    print("固有値を出力")
    print(eigvals)

    # 固有ベクトルを画像に変換して表示
    print("固有ベクトルを画像に変換して出力")
    for i in range(10):
        output_img(eigvecs[:, i], f"eigvec.png")

    # 潜在空間への射影
    print("潜在空間への射影")
    v = eigvecs[:, :DIM_L]
    y = v.T @ data

    # 多様体をプロット
    print("多様体をプロット")
    plot_manifold(y)
