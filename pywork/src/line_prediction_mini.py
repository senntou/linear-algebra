import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.const import DIM, DIM_L, M, N, R_RANGE, THETA_RANGE
from utils.input import get_input_images
from utils.input import get_input_images_params
from utils.input import get_rotation_matrix
from utils.calc import get_covariance_matrix, get_eigen, get_eigvals_of_lines
from utils.utils import output_img


MODEL_DIR = "model"
MODEL_NAME = MODEL_DIR + "/" + "line_prediction_mini.weights.h5"

NOISE_STD = 30


def save_model(model):  # モデルを保存
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_weights(MODEL_NAME)


def load_model():  # モデルを取得
    model = init_model()
    model.load_weights(MODEL_NAME)
    return model


def init_model():  # モデルの初期化・構築

    def custom_loss(y_true, y_pred):
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

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(DIM_L,)),

            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),

            # tf.keras.layers.Dense(
            #     16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            # tf.keras.layers.Dense(
            #     32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            # tf.keras.layers.Dense(
            #     16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005)),

            tf.keras.layers.Dense(2)
        ]
    )

    # model.compile(optimizer=tf.keras.optimizers.Adam(
    # learning_rate=0.001), loss=custom_loss)
    model.compile(optimizer='adam', loss=custom_loss)

    return model


def reset_model():  # モデルのリセット
    model = init_model()
    save_model(model)
    return model


def train_model(x_train, y_train, epochs=10):  # モデルの学習

    model = load_model()
    model.fit(x_train, y_train, epochs=epochs)
    save_model(model)
    return model


def evaluate_model(x_test, y_test):  # モデルの評価
    model = load_model()
    loss = model.evaluate(x_test, y_test)
    print("loss: ", loss)
    return loss


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


def get_eigvecs_diml():  # 固有ベクトルを取得
    eigvals, eigvecs = get_eigvals_of_lines(N, M)
    return eigvecs[:, :DIM_L]


def project_to_latent_space(x):  # 潜在空間への射影
    x = x.T
    v = get_eigvecs_diml()
    y = v.T @ x
    return y.T


def generate_data(n, m, times=1):  # データの生成

    x_base = get_input_images(n, m).T
    x_base = project_to_latent_space(x_base)
    y_base = get_input_images_params(n, m).T

    x = None
    y = None

    for _ in range(times):
        x_temp = x_base + np.random.normal(0, NOISE_STD, x_base.shape)

        if x is None:
            x = x_temp
            y = y_base
            continue
        else:
            x = np.vstack((x, x_temp))
            y = np.vstack((y, y_base))

    return x, y


def generate_random_data(num, latent=True):  # ランダムなデータを生成
    r_min = 0
    r_max = DIM / 2
    theta_min = 0
    theta_max = np.pi * 2

    data = np.zeros((num, DIM * DIM))
    params = np.zeros((2, num))
    params[0] = np.random.uniform(r_min, r_max, num)
    params[1] = np.random.uniform(theta_min, theta_max, num)

    for i in range(num):
        img = get_rotation_matrix(params[0, i], params[1, i])
        img = img.reshape(DIM * DIM)
        data[i] = img

    data += np.random.normal(0, NOISE_STD, data.shape)
    if latent:
        data = project_to_latent_space(data)

    return data, params.T


def main_printer(func):
    def wrapper():
        print("\n====== temp.py ======\n")
        func()
        print("\n======== end ========\n")
    return wrapper


def test_prediction():
    x_test, y_test = generate_random_data(1, latent=False)
    x_test_latent = project_to_latent_space(x_test)
    y_test = y_test[0]
    y_pred = predict_model(x_test_latent)[0]

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

    # train
    reset_model()
    x_train, y_train = generate_data(N, M)
    model = train_model(x_train, y_train, epochs=300)

    # evaluate
    x_eval, y_eval = generate_random_data(1000)
    model = load_model()
    model.evaluate(x_eval, y_eval)

    # test
    test_prediction()
