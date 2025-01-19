import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from cnn.cnn import LinePrediction
from utils.const import DIM, DIM_L, M, N, R_RANGE, THETA_RANGE
from utils.input import get_input_images_except_all_zero
from utils.input import get_rotation_matrix
from utils.calc import get_eigvals_of_lines


class LinePredictionSC(LinePrediction):
    MODEL_NAME = "sincos"

    def to_input(self, params):
        r_arr, theta_arr = params[:, 0], params[:, 1]
        sin = np.sin(theta_arr)
        cos = np.cos(theta_arr)
        return np.array([r_arr, sin, cos]).T

    def to_r_theta(self, params):
        r, sin, cos = params[:, 0], params[:, 1], params[:, 2]
        theta = np.arctan2(sin, cos)
        return np.array([r, theta]).T

    def init_model(self):  # モデルの初期化・構築
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(DIM_L,)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(3)
            ]
        )
        model.compile(optimizer='adam', loss='mse')
        return model
