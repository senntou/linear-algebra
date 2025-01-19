import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from cnn.cnn import LinePrediction
from utils.const import DIM, DIM_L, M, MODEL_DIR, N, R_RANGE, THETA_RANGE
from utils.input import get_input_images_except_all_zero
from utils.input import get_rotation_matrix
from utils.calc import get_eigvals_of_lines


class LinePredictionCL(LinePrediction):

    MODEL_NAME = "customloss"

    def custom_loss(self, y_true, y_pred):  # 損失関数
        r_true, theta_true = y_true[:, 0], y_true[:, 1]
        r_pred, theta_pred = y_pred[:, 0], y_pred[:, 1]
        r_weight = 1 / R_RANGE
        theta_weight = 1 / THETA_RANGE
        r_loss = tf.reduce_mean(tf.square(r_true - r_pred))

        pi = tf.constant(np.pi, dtype=tf.float32)
        theta_loss = tf.abs(theta_true - theta_pred)
        theta_loss = tf.minimum(tf.square(theta_loss),
                                tf.square(2 * pi - theta_loss))
        theta_loss = tf.reduce_mean(theta_loss)

        return r_weight * r_loss + theta_weight * theta_loss
