#include "utils.h"
#include "Eigen/src/Core/Matrix.h"
#include "cache.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <sys/stat.h>

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
  // input sum を確認
  std::string input_sum_dir = "cache/" + cache_id + "_input_sum.txt";
  std::ifstream ifs(input_sum_dir);

  // cacheがある場合
  if (ifs.is_open()) {
    std::string input_sum;
    ifs >> input_sum;
    double input_sum_double = std::stod(input_sum);
    // input sum が同じ場合、入力データが同じとみなしてcacheを読み込む
    if (abs(input.sum() - input_sum_double) < 1e-6) {
      std::cout << "[CACHE HIT] cache hit: " << cache_id << std::endl;
      load_vector_cache(eigvals, cache_id + "_eigvals");
      load_matrix_cache(eigvecs, cache_id + "_eigvecs");
      return;
    }
  }

  std::cout << "[CACHE MISS] cache miss: " << cache_id << std::endl;

  // input sumをcacheに書き込み
  char buffer[50];
  sprintf(buffer, "%.12f", input.sum());
  std::ofstream ofs(input_sum_dir);
  ofs << buffer << std::endl;

  // cacheがない場合、固有値、固有ベクトルを計算
  Eigen::EigenSolver<Eigen::MatrixXd> es(input);
  eigvals = es.eigenvalues().real();
  eigvecs = es.eigenvectors().real();

  // 固有値の大きい順に並び替え
  std::vector<std::pair<double, Eigen::VectorXd>> eig;
  for (int i = 0; i < DIM * DIM; i++) {
    eig.push_back({eigvals(i), eigvecs.col(i)});
  }
  std::sort(eig.begin(), eig.end(),
            ([](auto &lhs, auto &rhs) { return lhs.first > rhs.first; }));
  for (int i = 0; i < 8; i++) {
    eigvals(i) = eig[i].first;
    eigvecs.col(i) = eig[i].second;
  }

  // cacheに書き込み
  save_vector_cache(eigvals, cache_id + "_eigvals");
  save_matrix_cache(eigvecs, cache_id + "_eigvecs");
}
