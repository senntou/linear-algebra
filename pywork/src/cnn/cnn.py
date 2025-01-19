import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.const import DIM, DIM_L, M, MODEL_DIR, N, R_RANGE, THETA_RANGE
from utils.input import get_input_images_except_all_zero
from utils.input import get_rotation_matrix
from utils.calc import get_eigvals_of_lines


NOISE_STD = 30


class LinePrediction:

    MODEL_NAME = "default"
    MODEL_PATH = None

    def __init__(self):  # コンストラクタ
        print("##############################")
        print("## Line Prediction Instance ")
        print("## Model Name: ", self.MODEL_NAME)
        print("##############################")
        print()
        self.MODEL_PATH = MODEL_DIR + "/" + self.MODEL_NAME + ".weights.h5"

    def to_input(self, params):  # 入力データの変換
        return params

    def to_r_theta(self, params):  # 出力データの変換
        return params

    def custom_loss(self, y_true, y_pred):  # 損失関数
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def init_model(self):  # モデルの初期化・構築
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(DIM_L,)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(2)
            ]
        )

        # model.compile(optimizer=tf.keras.optimizers.Adam(
        # learning_rate=0.001), loss=custom_loss)
        model.compile(optimizer='adam', loss=self.custom_loss)

        return model

    def save_model(self, model):  # モデルを保存
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        model.save_weights(self.MODEL_PATH)

    def load_model(self):  # モデルを取得
        model = self.init_model()
        model.load_weights(self.MODEL_PATH)
        return model

    def reset_model(self):  # モデルのリセット
        model = self.init_model()
        self.save_model(model)
        return model

    def train_model(self, x_train, y_train, epochs=10):  # モデルの学習
        y_train = self.to_input(y_train)
        model = self.load_model()
        model.fit(x_train, y_train, epochs=epochs)
        self.save_model(model)
        return model

    def evaluate_model(self, x_test, y_test):  # モデルの評価
        y_test = self.to_input(y_test)
        model = self.load_model()
        loss = model.evaluate(x_test, y_test)
        print("loss: ", loss)
        return loss

    def show_graph_of_loss(self, history):
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train'], loc='upper left')
        plt.show()

    def predict_model(self, x_test):  # モデルを用いた予測
        model = self.load_model()
        result = model.predict(x_test)
        return self.to_r_theta(result)

    def get_eigvecs_diml(self):  # 固有ベクトルを取得
        eigvals, eigvecs = get_eigvals_of_lines(N, M)
        return eigvecs[:, :DIM_L]

    def project_to_latent_space(self, x):  # 潜在空間への射影
        x = x.T
        v = self.get_eigvecs_diml()
        y = v.T @ x
        return y.T

    def generate_data(self, n, m, times=1):  # データの生成
        y_base, x_base = get_input_images_except_all_zero(N, M)
        x_base = x_base.T
        y_base = y_base.T

        # x_baseをtimes回複製
        x = np.tile(x_base, (times, 1))
        y = np.tile(y_base, (times, 1))

        x = x + np.random.normal(0, NOISE_STD, x.shape)
        x = self.project_to_latent_space(x)

        return x, y

    def generate_random_data(self, num, latent=True):  # ランダムなデータを生成
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
            data = self.project_to_latent_space(data)

        return data, params.T

    def test_prediction(self):
        x_test, y_test = self.generate_random_data(1, latent=False)
        x_test_latent = self.project_to_latent_space(x_test)
        y_test = y_test[0]
        y_pred = self.predict_model(x_test_latent)[0]

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
        plt.savefig("save/" + self.MODEL_NAME + "_test.png")
