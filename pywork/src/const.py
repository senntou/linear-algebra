import numpy as np

DIM = 32
N = 100  # rの値の数
M = 100  # thetaの値の数

DIM_L = 3

BOLD = 2  # 入力画像の直線の太さ

R_MIN = 0
R_MAX = DIM / 2
R_RANGE = R_MAX - R_MIN
THETA_MIN = 0
THETA_MAX = np.pi * 2
THETA_RANGE = THETA_MAX - THETA_MIN
