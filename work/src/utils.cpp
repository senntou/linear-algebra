#include "utils.h"
#include "Eigen/src/Core/Matrix.h"
#include "serialize.h"
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
      img.at<double>(i, j) = data(i * DIM + j);
    }
  }
  myImWrite(filename, img);
}

// 固有値の計算
void eigensolve_cached(const Eigen::MatrixXd &input, Eigen::VectorXd &eigvals,
                       Eigen::MatrixXd &eigvecs, std::string cache_id) {

  // keyの生成
  string key = generate_key_by_Matrix(input);

  // キャッシュがあればそれを返す
  string values = get_cache(cache_id, key);
  if (values != "") {
    cout << "[CACHE HIT] eigensolve" << endl;
    pair<VectorXd, MatrixXd> output = deserialize_eigenvals(values);
    eigvals = output.first;
    eigvecs = output.second;
    return;
  }

  cout << "[CACHE MISS] eigensolve" << endl;

  EigenSolver<MatrixXd> es(input);
  VectorXd eigvals_tmp = es.eigenvalues().real();
  MatrixXd eigvecs_tmp = es.eigenvectors().real();

  // 固有値のソート
  vector<pair<double, int>> eigvals_index;
  for (int i = 0; i < eigvals_tmp.size(); i++) {
    eigvals_index.push_back(make_pair(eigvals_tmp(i), i));
  }
  sort(eigvals_index.begin(), eigvals_index.end(),
       greater<pair<double, int>>());
  VectorXd eigvals_sorted = VectorXd::Zero(eigvals_tmp.size());
  MatrixXd eigvecs_sorted =
      MatrixXd::Zero(eigvecs_tmp.rows(), eigvecs_tmp.cols());
  for (int i = 0; i < eigvals_tmp.size(); i++) {
    eigvals_sorted(i) = eigvals_tmp(eigvals_index[i].second);
    eigvecs_sorted.col(i) = eigvecs_tmp.col(eigvals_index[i].second);
  }

  // キャッシュに保存
  string serialized =
      serialize_eigenvals(make_pair(eigvals_sorted, eigvecs_sorted));
  save_cache(cache_id, key, serialized);

  eigvals = eigvals_sorted;
  eigvecs = eigvecs_sorted;

  return;
}
