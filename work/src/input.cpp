#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

#include "Eigen/src/Core/Matrix.h"
#include "utils.h"

using namespace std;
using namespace cv;
using namespace Eigen;

// i, j番目のr, thetaを取得
pair<double, double> get_r_theta(int i, int j) {
  double r_min = 0;
  double r_max = (double)DIM / 2;
  double theta_min = 0;
  double theta_max = 2 * M_PI;
  double r = r_min + (r_max - r_min) / (double)N * (double)i;
  double theta = theta_min + (theta_max - theta_min) / (double)M * (double)j;
  return make_pair(r, theta);
}

// 回転した直線の画像を生成
Mat get_rotation_matrix(double r, double theta) {
  const double BOLD = 2;
  Mat R = Mat::eye(DIM, DIM, CV_64F);
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      double x = i - (double)DIM / 2;
      double y = j - (double)DIM / 2;
      double dist = abs(x * cos(theta) + y * sin(theta) - r);
      double pix = clamp((BOLD - dist) * 255. / BOLD, 0., 255.);
      R.at<double>(i, j) = pix;
    }
  }
  return R;
}

vector<vector<VectorXd>> get_input_data() {
  // 画像の読み込み
  vector data(N, vector<VectorXd>(M));

  // r, thetaのstep数分だけ画像を生成
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {

      // i, j番目のr, thetaを取得
      pair<double, double> rt = get_r_theta(i, j);
      double r = rt.first;
      double theta = rt.second;
      Mat R = get_rotation_matrix(r, theta);
      MatrixXd temp = MatrixXd::Zero(DIM, DIM);
      cv2eigen(R, temp);
      data[i][j] = Map<VectorXd>(temp.data(), temp.size());
    }
  }

  return data;
}

// VectorXdに正規分布のノイズを加える
void add_noise(VectorXd &v, double sigma) {
  random_device seed_gen;
  default_random_engine engine(seed_gen());
  normal_distribution<> dist(0.0, sigma);

  for (int i = 0; i < v.size(); i++) {
    v(i) += dist(engine);
  }
}
