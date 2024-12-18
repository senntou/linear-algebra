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

  // 固有値の大きい順に並び替え
  vector<pair<double, VectorXd>> eig;
  for (int i = 0; i < DIM * DIM; i++) {
    eig.push_back({eigvals(i), eigvecs.col(i)});
  }
  sort(eig.begin(), eig.end(),
       ([](auto &lhs, auto &rhs) { return lhs.first > rhs.first; }));

  // 固有ベクトルの表示
  for (int i = 0; i < 8; i++) {
    VectorXd v = eig[i].second;
    show_heatmap(v, "eigvec", false);
  }

  // 潜在空間への射影
  const int DIM_L = 2; // 潜在空間の次元
  MatrixXd v = MatrixXd::Zero(DIM * DIM, DIM_L);
  for (int i = 0; i < DIM_L; i++) {
    v.col(i) = eig[i].second;
  }
  vector y(N, vector<VectorXd>(M));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      y[i][j] = v.transpose() * data[i][j];
    }
  }

  // 潜在空間の表示
  MatrixXd y0 = MatrixXd::Zero(N, M);
  MatrixXd y1 = MatrixXd::Zero(N, M);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      y0(i, j) = y[i][j](0);
      y1(i, j) = y[i][j](1);
    }
  }
  show_heatmap(y0, "latent", false);
  show_heatmap(y1, "latent", false);
}
