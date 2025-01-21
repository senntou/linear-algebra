
from cnn.cnn import LinePrediction
import numpy as np


NOISE_STD = 0


class LinePredictionVector(LinePrediction):

    MODEL_NAME = "vector"

    def to_input(self, params):
        r_arr, theta_arr = params[:, 0], params[:, 1]
        x = r_arr * np.cos(theta_arr)
        y = r_arr * np.sin(theta_arr)
        return np.array([x, y]).T

    def to_r_theta(self, params):
        x_arr, y_arr = params[:, 0], params[:, 1]
        r = np.sqrt(x_arr**2 + y_arr**2)
        theta = np.arctan2(y_arr, x_arr)
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)
        return np.array([r, theta]).T
