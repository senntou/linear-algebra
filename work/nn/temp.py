import tensorflow as tf
import numpy as np

NUM = 10000

# 16次元ベクトルを下に、2つのパラメータを回帰する
# DNNを使って、2つのパラメータを回帰する
# 16の入力から、平均と分散を出力する


def get_data():
    # 学習データの生成
    x_train = []
    y_train = []
    for _ in range(NUM):
        ave = np.random.randint(0, 100)
        var = np.random.randint(0, 100)
        x_train.append(np.random.normal(ave, var, 16))
        y_train.append([ave, var])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train


def main_printer(func):

    def wrapper():
        print("\n====== temp.py ======\n")
        func()
        print("\n======== end ========\n")

    return wrapper


@main_printer
def main():
    x_train, y_train = get_data()

    # モデルの構築
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(16,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(2)
        ]
    )

    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=10)


if __name__ == "__main__":
    main()
