import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.const import DIM, DIM_L, M, N, R_RANGE, THETA_RANGE
from utils.input import get_input_images_except_all_zero
from utils.input import get_rotation_matrix
from utils.calc import get_eigvals_of_lines


MODEL_DIR = "model"
MODEL_NAME = MODEL_DIR + "/" + "line_prediction_mini.weights.h5"

NOISE_STD = 30


def to_sincos(params):
    r_arr, theta_arr = params[:, 0], params[:, 1]
    sin = np.sin(theta_arr)
    cos = np.cos(theta_arr)

    return np.array([r_arr, sin, cos]).T


def to_r_theta(params):
    r, sin, cos = params[:, 0], params[:, 1], params[:, 2]
    theta = np.arctan2(sin, cos)

    return np.array([r, theta]).T


def save_model(model):  # モデルを保存
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_weights(MODEL_NAME)


def load_model():  # モデルを取得
    model = init_model()
    model.load_weights(MODEL_NAME)
    return model


def init_model():  # モデルの初期化・構築

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

    # model.compile(optimizer=tf.keras.optimizers.Adam(
    # learning_rate=0.001), loss=custom_loss)
    model.compile(optimizer='adam', loss='mse')

    return model


def reset_model():  # モデルのリセット
    model = init_model()
    save_model(model)
    return model


def train_model(x_train, y_train, epochs=10):  # モデルの学習

    y_train = to_sincos(y_train)

    model = load_model()
    model.fit(x_train, y_train, epochs=epochs)
    save_model(model)
    return model


def evaluate_model(x_test, y_test):  # モデルの評価

    y_test = to_sincos(y_test)

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
    result = model.predict(x_test)

    # return result
    return to_r_theta(result)


def get_eigvecs_diml():  # 固有ベクトルを取得
    eigvals, eigvecs = get_eigvals_of_lines(N, M)
    return eigvecs[:, :DIM_L]


def project_to_latent_space(x):  # 潜在空間への射影
    x = x.T
    v = get_eigvecs_diml()
    y = v.T @ x
    return y.T


def generate_data(n, m, times=1):  # データの生成

    y_base, x_base = get_input_images_except_all_zero(N, M)
    x_base = x_base.T
    y_base = y_base.T

    # x_baseをtimes回複製
    x = np.tile(x_base, (times, 1))
    y = np.tile(y_base, (times, 1))

    x = x + np.random.normal(0, NOISE_STD, x.shape)
    x = project_to_latent_space(x)

    return x, y


def generate_random_data(num, latent=True):  # ランダムなデータを生成
    r_min = 0
    # r_max = DIM / 2
    r_max = DIM * np.sqrt(1 / 2)
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
    # plt.show()
    plt.savefig("save/line_prediction_mini.png")


@main_printer
def main():
    TRAIN = True
    # TRAIN = False
    if TRAIN:
        # train
        reset_model()
        x_train, y_train = generate_data(N, M)
        print(y_train.shape)
        model = train_model(x_train, y_train, epochs=100)

        # evaluate
        x_eval, y_eval = generate_random_data(1000, True)
        model = load_model()
        y_eval = to_sincos(y_eval)
        model.evaluate(x_eval, y_eval)

    # test
    test_prediction()


if __name__ == '__main__':
    main()
    # sub()
