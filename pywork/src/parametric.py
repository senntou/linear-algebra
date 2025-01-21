from utils.utils import output_img
import matplotlib.pyplot as plt
from utils.const import DIM, DIM_L, N, M
from utils.input import get_input_images_except_all_zero
from utils.calc import get_eigvals_of_lines


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
    _params, data = get_input_images_except_all_zero(N, M)  # 各列が画像のベクトル

    # 固有値と固有ベクトルを計算
    print("固有値と固有ベクトルを計算")
    eigvals, eigvecs = get_eigvals_of_lines(N, M)

    # 固有値を出力
    print("固有値を出力")
    print(eigvals)

    # 固有ベクトルを画像に変換して表示
    # print("固有ベクトルを画像に変換して出力")
    # for i in range(10):
    #     output_img(eigvecs[:, i], f"eigvec.png")

    # 1~4番目の固有ベクトルを1枚の画像に変換して出力
    print("1~4番目の固有ベクトルを1枚の画像に変換して出力")
    plt.subplot(2, 2, 1)
    plt.title("1st")
    plt.imshow(eigvecs[:, 0].reshape(DIM, DIM), cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title("2nd")
    plt.imshow(eigvecs[:, 1].reshape(DIM, DIM), cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title("3rd")
    plt.imshow(eigvecs[:, 2].reshape(DIM, DIM), cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title("4th")
    plt.imshow(eigvecs[:, 3].reshape(DIM, DIM), cmap='gray')

    plt.savefig("output/eigvecs.png", bbox_inches='tight')
    plt.close()

    # 潜在空間への射影
    print("潜在空間への射影")
    v = eigvecs[:, :DIM_L]
    y = v.T @ data

    # 多様体をプロット
    # except_zeroの場合はうまくうごかない
    # print("多様体をプロット")
    # plot_manifold(y)
