#pragma once
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

#include "Eigen/src/Core/Matrix.h"
#include "utils.h"

using namespace std;
using namespace cv;
using namespace Eigen;

vector<vector<VectorXd>> get_input_data() {
  // 画像の読み込み
  vector data(N, vector<VectorXd>(M));

  // r, thetaの範囲
  double r_min = 0;
  double r_max = (double)DIM / 2;
  double theta_min = 0;
  double theta_max = 2 * M_PI;
  // r, thetaのstep数分だけ画像を生成
  for (int i = 0; i < N; i++) {
    double r = r_min + (r_max - r_min) / (double)N * (double)i;
    for (int j = 0; j < M; j++) {
      double theta =
          theta_min + (theta_max - theta_min) / (double)M * (double)j;
      Mat R = get_rotation_matrix(r, theta);
      MatrixXd temp = MatrixXd::Zero(DIM, DIM);
      cv2eigen(R, temp);
      data[i][j] = Map<VectorXd>(temp.data(), temp.size());
    }
  }

  return data;
}
