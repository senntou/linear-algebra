#include "utils.h"
#include "Eigen/src/Core/Matrix.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <sys/stat.h>

using namespace std;
using namespace Eigen;

// 画像の保存
void myImWrite(std::string filename, const cv::Mat &img) {
  static std::map<std::string, int> counter;
  std::string output_dir = "output/" + filename + "_";
  output_dir += std::to_string(++counter[filename]) + ".png";

  cv::imwrite(output_dir, img);
}

// matrixをCSVに保存
void save_matrix_to_csv(const std::string &filename,
                        const Eigen::MatrixXd &matrix) {
  std::ofstream file("pydata/" + filename + ".csv");
  if (!file.is_open()) {
    std::cerr << "[ERROR] cannot open file" << std::endl;
    return;
  }

  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      file << matrix(i, j);
      if (j < matrix.cols() - 1) {
        file << ","; // カンマで区切る
      }
    }
    file << "\n"; // 行の終わりで改行
  }

  file.close();
}

// ヒートマップの表示
void show_heatmap(const Eigen::MatrixXd &data_input, std::string filename,
                  bool max_255) {
  Eigen::MatrixXd data = data_input;
  if (!max_255) {
    data = data.array() - data.minCoeff();
    data = data * 255 / data.maxCoeff();
  }

  const int ROWS = data.rows();
  const int COLS = data.cols();
  cv::Mat img(ROWS, COLS, CV_64F);
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      img.at<double>(i, j) = data(i, j);
    }
  }
  myImWrite(filename, img);
}
void show_heatmap(const Eigen::VectorXd &data_input, std::string filename,
                  bool max_255) {
  Eigen::VectorXd data = data_input;
  if (!max_255) {
    data = data.array() - data.minCoeff();
    data = data * 255 / data.maxCoeff();
  }

  assert("utils.cpp:44" && data.size() == DIM * DIM);
  cv::Mat img(DIM, DIM, CV_64F);
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      img.at<double>(i, j) = data(j * DIM + i);
    }
  }
  myImWrite(filename, img);
}
