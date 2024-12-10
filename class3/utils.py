import numpy as np
import matplotlib.pyplot as plt

DIM = 8
DIR = "output/"
SPLIT = 10


def get_x_base(flatten=False):
    vec = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    x_base = []

    for i in range(DIM):
        xx = np.zeros((DIM, DIM))
        xx[:, i] = vec
        x_base.append(xx)

    if flatten:
        return [x.flatten() for x in x_base]

    return x_base


def unflatten(x):
    return x.reshape(DIM, DIM)


def get_x_pnt(x_base, flatten=False):
    x_pnt = []
    for i in range(DIM - 1):
        for lmd in np.linspace(0, 1, SPLIT + 1)[:-1]:
            x_pnt.append(x_base[i] * (1 - lmd) + x_base[i + 1] * lmd)

    return x_pnt


def show_heatmap(x, title="", save=False):
    plt.imshow(x, cmap="bone")
    plt.title(title)
    if save:
        plt.savefig(DIR + title + ".png")
        plt.close()
    else:
        plt.show()


# 4つの固有ベクトルを表示
def show_heatmaps(x_list, title="", save=False):
    fig = plt.figure(figsize=(8, 8))
    for i, x in enumerate(x_list):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(x, cmap="bone")
        ax.set_title(f"{title} {i + 1}")

    if save:
        plt.savefig(DIR + title + ".png")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_3d(x, y, z, title="", save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", title=title)
    ax.plot(x, y, z, marker="o")
    if save:
        plt.savefig(DIR + title + ".png")
        plt.close()
    plt.show()


def get_input_image_vector(line=1):
    if not 1 <= line <= DIM:
        raise ValueError("line must be between 1 and 8")
    x = np.zeros((DIM, DIM))
    x[:, line - 1] = np.ones(DIM)

    noise = np.random.normal(0, 0.1, (DIM, DIM))

    return (x + noise).flatten()
