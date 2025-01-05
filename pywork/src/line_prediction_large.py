import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from const import DIM
from input import get_input_images
from input import get_input_images_params
from input import get_rotation_matrix


NUM = 10000
MODEL_DIR = "model"
MODEL_NAME = MODEL_DIR + "/" + "line_prediction.keras"

NOISE_STD = 30


def save_model(model):  # モデルを保存
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save(MODEL_NAME)


def load_model():  # モデルを取得
    model = tf.keras.models.load_model(MODEL_NAME)
    return model


def init_model():  # モデルの初期化・構築
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(DIM * DIM,)),

            tf.keras.layers.Dense(
                64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.Dense(
                32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.Dense(
                16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005)),

            tf.keras.layers.Dense(2)
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001), loss="mse")
    save_model(model)


def train_model(x_train, y_train, epochs=10):  # モデルの学習
    model = load_model()
    model.fit(x_train, y_train, epochs=epochs)
    save_model(model)
    return model


def show_graph_of_loss(history):
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train'], loc='upper left')
    plt.show()


def predict_model(x_test):  # モデルを用いた予測
    model = load_model()
    return model.predict(x_test)


def generate_data(n, m, times=1):  # データの生成

    x_base = get_input_images(n, m).T
    y_base = get_input_images_params(n, m).T

    x = None
    y = None

    for _ in range(times):
        x_temp = x_base + np.random.normal(0, NOISE_STD, x_base.shape)
        x_temp = np.clip(x_temp, 0, 255)

        if x is None:
            x = x_temp
            y = y_base
            continue
        else:
            x = np.vstack((x, x_temp))
            y = np.vstack((y, y_base))

    return x, y


def generate_random_data(NUM):  # ランダムなデータを生成
    r_min = 0
    r_max = DIM / 2
    theta_min = 0
    theta_max = np.pi * 2

    data = np.zeros((NUM, DIM * DIM))
    params = np.zeros((2, NUM))
    params[0] = np.random.uniform(r_min, r_max, NUM)
    params[1] = np.random.uniform(theta_min, theta_max, NUM)

    for i in range(NUM):
        img = get_rotation_matrix(params[0, i], params[1, i])
        img = img.reshape(DIM * DIM)
        data[i] = img

    noise = np.random.normal(0, NOISE_STD, data.shape)
    data = np.array(data) + noise

    return data, params.T


def main_printer(func):
    def wrapper():
        print("\n====== temp.py ======\n")
        func()
        print("\n======== end ========\n")
    return wrapper


def test_prediction():
    x_test, y_test = generate_random_data(1)
    y_test = y_test[0]
    y_pred = predict_model(x_test)[0]

    img_pred = get_rotation_matrix(y_pred[0], y_pred[1])

    print("test: ", y_test)
    print("predict: ", y_pred)

    plt.subplot(1, 2, 1)
    plt.imshow(x_test.reshape(DIM, DIM), cmap='gray')
    plt.title("test")
    plt.subplot(1, 2, 2)
    plt.imshow(img_pred, cmap='gray')
    plt.title("predict")
    plt.show()


if __name__ == '__main__':

    # evaluate
    x_eval, y_eval = generate_random_data(1000)
    model = load_model()
    model.evaluate(x_eval, y_eval)

    # test
    test_prediction()
