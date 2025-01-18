from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils.calc import get_eigvals_of_lines
from utils.const import BIG_IMAGE_HEIGHT, BIG_IMAGE_WIDTH, DIM, M, N
from utils.input import get_rotation_matrix
from line_prediction_mini import DIM_L, predict_model, project_to_latent_space

WIDTH = BIG_IMAGE_WIDTH
HEIGHT = BIG_IMAGE_HEIGHT

IMG_PATH = "input/1.png"


if __name__ == '__main__':

    # 画像の読み込み
    img = Image.open(IMG_PATH)
    img = np.array(img)

    # DIMxDIMのパッチに分割
    h = img.shape[0] // DIM
    w = img.shape[1] // DIM
    patches = np.zeros((h, w, DIM, DIM), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            patches[i, j] = img[i*DIM:(i+1)*DIM, j*DIM:(j+1)*DIM]

    # 全ての輝度値が0の画像を検出
    invalid_patches_flg = np.all(patches == 0, axis=(2, 3))

    # 有効画像をモデルにより置換
    valid_patches = patches[~invalid_patches_flg]
    valid_patches = valid_patches.reshape(-1, DIM * DIM)
    _, eigvecs = get_eigvals_of_lines(N, M)

    v = eigvecs[:, :DIM_L]

    valid_patches_latent = (v.T @ valid_patches.T).T
    print(valid_patches_latent.shape)

    params = predict_model(valid_patches_latent)

    valid_patches_reconstructed = np.zeros_like(valid_patches)
    for i in range(valid_patches.shape[0]):
        valid_patches_reconstructed[i] = get_rotation_matrix(
            params[i, 0], params[i, 1]).reshape(-1)

    img_reconstructed = np.zeros_like(img)
    idx = 0
    for i in range(h):
        for j in range(w):
            if invalid_patches_flg[i, j]:
                continue
            img_reconstructed[i*DIM:(i+1)*DIM, j*DIM:(j+1) *
                              DIM] = valid_patches_reconstructed[idx].reshape(DIM, DIM)
            idx += 1

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("original")
    plt.grid(which='both', color='r', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(0, img.shape[1], DIM), color='gray')
    plt.yticks(np.arange(0, img.shape[0], DIM), color='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(img_reconstructed, cmap='gray')
    plt.title("reconstructed")
    plt.grid(which='both', color='r', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(0, img_reconstructed.shape[1], DIM))
    plt.yticks(np.arange(0, img_reconstructed.shape[0], DIM))

    plt.savefig("save/prparams_2.png")

    plt.show()
