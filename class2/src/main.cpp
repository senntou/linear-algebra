#include "opencv2/core/mat.hpp"
#include "utils.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cwchar>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <string>

#define OUTPUT_DIR "output/"

#define M 28 // yの次元
#define K 5  // hの次元
const int N = M - K + 1;

Eigen::Vector<double, M * M> cv2eigen(cv::Mat &img) {
  Eigen::Vector<double, M * M> x;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      x(i * M + j) = img.at<uchar>(i, j);
    }
  }
  return x;
}

cv::Mat eigen2cv(const Eigen::Vector<double, M * M> &y) {
  cv::Mat img = cv::Mat::zeros(M, M, CV_8UC1);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      img.at<uchar>(i, j) = std::clamp(y(i * M + j), 0.0, 255.0);
    }
  }
  return img;
}

// f = 1/2 * ||y - Ax||^2 に対し、fの微分を計算する関数
Eigen::Vector<double, M * M> grad_f(const Eigen::Vector<double, N * N> &y,
                                    const Eigen::MatrixXd &A,
                                    const Eigen::Vector<double, M * M> &x) {
  return A.transpose() * (A * x - y);
}

// 射影勾配法
Eigen::Vector<double, M * M>
projection_gradient_method(const Eigen::Vector<double, N * N> &y,
                           const Eigen::MatrixXd &A, const double mu,
                           const int max_iter) {
  Eigen::Vector<double, M * M> x = Eigen::Vector<double, M * M>::Zero();
  for (int i = 0; i < M * M; i++) {
    x(i) = 10000;
  }

  for (int i = 0; i < max_iter; i++) {
    x -= mu * grad_f(y, A, x);
    for (int j = 0; j < M * M; j++) {
      x(j) = std::max(0.0, x(j));
    }
  }
  return x;
}

double sign(double x) { return (x > 0) - (x < 0); }

// 近接勾配法
Eigen::Vector<double, M * M>
proximal_gradient_method(const Eigen::Vector<double, N * N> &y,
                         const Eigen::MatrixXd &A, const double mu,
                         const double lambda, const int max_iter) {
  Eigen::Vector<double, M * M> x = Eigen::Vector<double, M * M>::Zero();
  for (int i = 0; i < M * M; i++) {
    x(i) = 0;
  }

  for (int i = 0; i < max_iter; i++) {
    x -= mu * grad_f(y, A, x);
    for (int j = 0; j < M * M; j++) {
      x(j) = sign(x(j)) * std::max(0.0, std::abs(x(j)) - lambda);
    }
  }
  return x;
}

int main() {
  // 画像読み込み
  std::string fileName = "mnist_8_024.png";
  cv::Mat img = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
  std::cout << "img size: " << img.size() << std::endl;

  // Mat型からEigen::Vector型に変換 (1次元ベクトルへ)
  Eigen::Vector<double, M * M> x_base = cv2eigen(img);

  // フィルタ
  Eigen::Matrix<double, K, K> h = Eigen::Matrix<double, K, K>::Ones();
  Eigen::Vector<double, M * M> a = Eigen::Vector<double, M * M>::Zero();
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < K; j++) {
      a(i * M + j) = h(i, j);
    }
  }

  // Aはhで畳み込み演算を行うための行列
  Eigen::MatrixXd A(N * N, M * M);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A.row(i * N + j) = a;
      vector_right_shift(a, 1);
      if (j == N - 1) {
        vector_right_shift(a, K - 1);
      }
    }
  }

  Eigen::Vector<double, N * N> y = A * x_base;

  // 射影勾配法
  // muの値を変えつつ実行
  const int max_iter = 1000;
  for (int i = 0; i < 10; i++) {
    double mu = 0.001 * (double)(i + 1);
    Eigen::Vector<double, M * M> x =
        projection_gradient_method(y, A, mu, max_iter);
    // 復元画像
    cv::Mat output = eigen2cv(x);
    myImWrite("projection", output);
  }

  // 近接勾配法
  // muの値を変えつつ実行
  for (int i = 0; i < 10; i++) {
    double mu = 0.001 * (double)(i + 1);
    double lambda = 1;
    Eigen::Vector<double, M * M> x =
        proximal_gradient_method(y, A, mu, lambda, max_iter);

    // 復元画像
    cv::Mat output = eigen2cv(x);
    myImWrite("proximal", output);
  }
  return 0;
}
