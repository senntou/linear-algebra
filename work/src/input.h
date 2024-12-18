#pragma once
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

#include "Eigen/src/Core/Matrix.h"

using namespace std;
using namespace cv;
using namespace Eigen;

// 回転した直線の画像を生成
Mat get_rotation_matrix(double r, double theta);

// データの生成と読み込み
vector<vector<VectorXd>> get_input_data();

// VectorXdに正規分布のノイズを加える
void add_noise(VectorXd &v, double sigma);
