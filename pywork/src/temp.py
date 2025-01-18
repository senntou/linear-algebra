import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.const import DIM, M, N
from utils.input import get_input_images_except_all_zero

NUM = 10000


def main_printer(func):
    def wrapper():
        print("\n====== temp.py ======\n")
        func()
        print("\n======== end ========\n")
    return wrapper


@main_printer
def main():
    _params, data = get_input_images_except_all_zero(N, M)
    data = data.T
    print(data.shape)
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        plt.imshow(data[i * 80].reshape(DIM, DIM), cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
