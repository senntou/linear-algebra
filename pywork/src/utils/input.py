import numpy as np
from utils.cache import sqlite_cache
from utils.const import DIM, BOLD, R_MAX, R_MIN, THETA_MAX, THETA_MIN


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
    data = []
    for i in range(N):
        for j in range(M):
            r = R_MIN + (R_MAX - R_MIN) * i / N
            theta = THETA_MIN + (THETA_MAX - THETA_MIN) * j / M
            img = get_rotation_matrix(r, theta)
            img = img.reshape(DIM * DIM)
            data.append(img)
    return np.array(data).T


@sqlite_cache("input_except_all_zero.db")
def get_input_images_except_all_zero(N, M):
    data = get_input_images(DIM, N, M).T
    data = data[~np.all(data == 0, axis=1)]
    return data.T


def get_input_images_params(N, M):
    params = []
    for i in range(N):
        for j in range(M):
            r = R_MIN + (R_MAX - R_MIN) * i / N
            theta = THETA_MIN + (THETA_MAX - THETA_MIN) * j / M
            params.append(np.array([r, theta]))
    params = np.array(params)
    return params.T
