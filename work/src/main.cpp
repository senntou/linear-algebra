#define WITHOUT_NUMPY 1
#include "../../../matplotlibcpp.h"
#include "input.h"
#include "utils.h"
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

int main() {
  // 画像の読み込み
  vector data = get_input_data();

  // 平均ベクトル
  MatrixXd x_mean = VectorXd::Zero(DIM * DIM);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      x_mean += data[i][j];
    }
  }
  x_mean /= N * M;

  // 分散共分散行列
  MatrixXd sigma = MatrixXd::Zero(DIM * DIM, DIM * DIM);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      VectorXd x_norm = data[i][j] - x_mean;
      sigma += data[i][j] * data[i][j].transpose();
    }
  }
  sigma /= N * M;
  show_heatmap(sigma, "covariance");

  // 固有値、固有ベクトル
  VectorXd eigvals;
  MatrixXd eigvecs;
  eigensolve_cached(sigma, eigvals, eigvecs, "eigensolve");

  // 固有ベクトルの表示
  for (int i = 0; i < 8; i++) {
    VectorXd v = eigvecs.col(i);
    show_heatmap(v, "eigvec", false);
  }

  // 潜在空間への射影
  const int DIM_L = 2; // 潜在空間の次元
  MatrixXd v = eigvecs.block(0, 0, DIM * DIM, DIM_L);
  vector y(N, vector<VectorXd>(M));
  MatrixXd y0 = MatrixXd::Zero(N, M);
  MatrixXd y1 = MatrixXd::Zero(N, M);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      y[i][j] = v.transpose() * data[i][j];
      y0(i, j) = y[i][j](0);
      y1(i, j) = y[i][j](1);
    }
  }

  show_heatmap(y0, "latent", false);
  show_heatmap(y1, "latent", false);

  // 多様体の可視化
  vector<double> y0_vec, y1_vec;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      y0_vec.push_back(y0(i, j));
      y1_vec.push_back(y1(i, j));
    }
  }
  matplotlibcpp::scatter(y0_vec, y1_vec);
  matplotlibcpp::show();

  // 観測データからの復元
  Mat x_cvmat = get_rotation_matrix(1.3, M_PI / 8.);
  MatrixXd x_eigmat = MatrixXd::Zero(DIM, DIM);
  cv2eigen(x_cvmat, x_eigmat);
  VectorXd x_input = Map<VectorXd>(x_eigmat.data(), x_eigmat.size());
  add_noise(x_input, 50);

  show_heatmap(x_input, "input", false);

  // 潜在空間への射影
  VectorXd y_input = v.transpose() * x_input;
}
