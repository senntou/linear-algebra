import numpy as np
from cache import sqlite_cache
from const import DIM, BOLD


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


def get_input_images_params(N, M):
    r_min = 0
    r_max = DIM / 2
    theta_min = 0
    theta_max = np.pi * 2
    params = []
    for i in range(N):
        for j in range(M):
            r = r_min + (r_max - r_min) * i / N
            theta = theta_min + (theta_max - theta_min) * j / M
            params.append(np.array([r, theta]))
    params = np.array(params)
    return params.T
