import os
import matplotlib.pyplot as plt
from utils.const import DIM

# filename => id のmap
id_map = {}


def output_img(img, filename, dir="output"):

    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.imshow(img.reshape(DIM, DIM), cmap="gray")

    id = id_map.get(filename)
    if id is None:
        id = 1

    plt.savefig(f"{dir}/{filename}_{id}.png")
    plt.close()

    id_map[filename] = id + 1
