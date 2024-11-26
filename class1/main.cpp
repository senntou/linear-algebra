#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <random>
#include <string>

#include "utils.h"

#define N 5

int main() {

  // (1) 畳み込み演算
  Eigen::Vector<double, N> b, x, h;
  x << 1, 2, 3, 4, 5;
  h << 2, 1, 0, 0, 1;

  for (int i = 0; i < N; i++) {
    b(i) = x.dot(h);
    vector_shift(h, 1);
  }

  print_vector("b", b);

  // (2) テプリッツ行列を用いた畳み込み演算
  Eigen::Matrix<double, N, N> A = Eigen::Matrix<double, N, N>::Zero();

  for (int i = 0; i < N; i++) {
    A.row(i) = h;
    vector_shift(h, 1);
  }

  print_matrix("A", A);
  b = A * x;
  print_vector("b", b);

  // (3) 信号に正規分布に基づくノイズを加える
  Eigen::Vector<double, N> x_noisy = x;
  double sigma = 0.1;
  Eigen::Vector<double, N> noise;

  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());

  // 平均0.0、標準偏差1.0で分布させる
  std::normal_distribution<> dist(0.0, sigma);

  for (int i = 0; i < N; i++) {
    noise(i) = dist(engine);
  }
  print_vector("noise", noise);
  b = A * x + noise;
  print_vector("b", b);

  // (4) 信号の復元
  // 回帰
  Eigen::Vector<double, N> x_hat =
      (A.transpose() * A).inverse() * A.transpose() * b;
  print_vector("x_hat", x_hat);

  // リッジ回帰
  double lambda = 0.1;
  x_hat = (A.transpose() * A + lambda * Eigen::Matrix<double, N, N>::Identity())
              .inverse() *
          A.transpose() * b;

  print_vector("x_hat", x_hat);

  return 0;
}
