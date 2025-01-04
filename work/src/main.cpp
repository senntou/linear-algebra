#include "calc.h"
#include "input.h"
#include "utils.h"
#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <iostream>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

// 画像は32x32のグレースケール画像
int main() {
  // 画像の読み込み
  vector data = get_input_data();

  MatrixXd sigma = calc_covariance_cached(data);

  show_heatmap(sigma, "covariance");

  // 固有値、固有ベクトル
  VectorXd eigvals;
  MatrixXd eigvecs;
  eigensolve_cached(sigma, eigvals, eigvecs);

  // 固有ベクトルの表示
  for (int i = 0; i < 8; i++) {
    VectorXd v = eigvecs.col(i);
    show_heatmap(v, "eigvec", false);
  }

  // 潜在空間への射影
  const int DIM_L = 3; // 潜在空間の次元
  MatrixXd v = eigvecs.block(0, 0, DIM * DIM, DIM_L);
  vector y(N, vector<VectorXd>(M));
  MatrixXd y0 = MatrixXd::Zero(N, M);
  MatrixXd y1 = MatrixXd::Zero(N, M);
  MatrixXd y2 = MatrixXd::Zero(N, M);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      y[i][j] = v.transpose() * data[i][j];
      y0(i, j) = y[i][j](0);
      y1(i, j) = y[i][j](1);
      y2(i, j) = y[i][j](2);
    }
  }

  save_matrix_to_csv("y0", y0);
  save_matrix_to_csv("y1", y1);
  save_matrix_to_csv("y2", y2);

  // 観測データからの復元
  Mat x_cvmat = get_rotation_matrix(1.3, M_PI / 8.);
  MatrixXd x_eigmat = MatrixXd::Zero(DIM, DIM);
  cv2eigen(x_cvmat, x_eigmat);

  VectorXd x_input = Map<VectorXd>(x_eigmat.data(), x_eigmat.size());
  add_noise(x_input, 50);

  show_heatmap(x_input, "input", false);

  // 潜在空間への射影
  VectorXd y_input = v.transpose() * x_input;

  // r, thetaの推定
  double r_input = 0;
  double theta_input = 0;
  double min_dist = 1e9;
  // y の中から、最もy_input に近いものを探す
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      double dist = (y_input - y[i][j]).norm();
      if (dist < min_dist) {
        min_dist = dist;
        r_input = get_r_theta(i, j).first;
        theta_input = get_r_theta(i, j).second;
      }
    }
  }

  cout << "r_input: " << r_input << endl;
  cout << "theta_input: " << theta_input / M_PI << "pi" << endl;

  // 復元画像の表示
  Mat x_reconstructed = get_rotation_matrix(r_input, theta_input);
  MatrixXd x_reconstructed_eigmat;
  cv2eigen(x_reconstructed, x_reconstructed_eigmat);
  show_heatmap(x_reconstructed_eigmat, "reconstructed", false);
}
