import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from cnn.cnn import LinePrediction
from cnn.line_prediction_mini import LinePredictionCL
from cnn.line_prediction_sincos import LinePredictionSC
from utils.const import DIM, M, N
from utils.input import get_input_images_except_all_zero
from utils.predict_params import predict_params_from_image

NUM = 10000


def main_printer(func):
    def wrapper():
        print("\n====== temp.py ======\n")
        func()
        print("\n======== end ========\n")
    return wrapper


@main_printer
def main():
    # instance
    lp = LinePrediction()

    # train
    lp.reset_model()
    x_train, y_train = lp.generate_data(N, M)
    model = lp.train_model(x_train, y_train, epochs=100)

    # evaluate
    x_test, y_test = lp.generate_random_data(1000)
    lp.evaluate_model(x_test, y_test)

    # test
    lp.test_prediction()
    predict_params_from_image(lp)


if __name__ == "__main__":
    main()
