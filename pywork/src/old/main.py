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
    plt.savefig("manifold.png")
    plt.close()


if __name__ == '__main__':
    print("\n=== main.py ===")

    # 画像の読み込み
    print("画像の読み込み")
    data = get_input_images(N, M)  # 各列が画像のベクトル

    # 固有値と固有ベクトルを計算
    print("固有値と固有ベクトルを計算")
    eigvals, eigvecs = get_eigvals_of_lines(N, M)

    # 固有ベクトルを画像に変換して表示
    # print("固有ベクトルを画像に変換して出力")
    # for i in range(10):
    #     output_img(eigvecs[:, i], f"eig{i}.png")

    # 潜在空間への射影
    print("潜在空間への射影")
    v = eigvecs[:, :DIM_L]
    y = v.T @ data

    # 多様体をプロット
    print("多様体をプロット")
    plot_manifold(y)

    # 観測データを擬似的に生成
    print("観測データを擬似的に生成")
    obs_data = get_rotation_matrix(2, np.pi / 8).reshape(DIM * DIM)
    obs_data += np.random.normal(0, 30, obs_data.shape)

    # 観測データを潜在空間に射影
    obs_data_latent = v.T @ obs_data

    # r, thetaの値を予測
    prediction = predict_model(np.array([obs_data_latent]))
    print("予測結果")
    print(f"r: {prediction[0][0]}, theta: {prediction[0][1]}")
    print("正解値")
    print(f"r: 2, theta: {np.pi / 8}")

    # 画像を復元
    print("画像を復元")
    img_pred = get_rotation_matrix(prediction[0][0], prediction[0][1])
    img_obs = obs_data.reshape(DIM, DIM)

    # 画像を表示
    print("画像を表示")
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_obs, cmap="gray")
    ax[0].set_title("Observation")
    ax[1].imshow(img_pred, cmap="gray")
    ax[1].set_title("Prediction")
    plt.show()
