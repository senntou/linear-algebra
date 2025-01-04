import os
import matplotlib.pyplot as plt
from const import DIM


def output_img(img, filename, dir="output"):

    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.imshow(img.reshape(DIM, DIM), cmap="gray")

    id = 1
    while True:
        if not os.path.exists(f"{filename}_{id}.png"):
            break
        id += 1

    plt.savefig(f"{dir}/{filename}_{id}.png")
    plt.close()
