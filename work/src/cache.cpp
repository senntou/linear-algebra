
#include "cache.h"
#include "Eigen/src/Core/Matrix.h"
#include <cstdio>
#include <fstream>

// Eigen::VectorXdをcacheファイルに書き込む
void save_vector_cache(const Eigen::VectorXd data, std::string cache_id) {
  std::string output_dir = "cache/" + cache_id + ".txt";
  std::ofstream ofs(output_dir);

  ofs << data.size() << "\n";

  for (int i = 0; i < data.size(); i++) {
    char buffer[50];
    sprintf(buffer, "%.12lf", data(i));
    std::string data_str = buffer;
    ofs << data_str << "\n";
  }
}

// cacheファイルからEigen::VectorXdを読み込む
void load_vector_cache(Eigen::VectorXd &dst, std::string cache_id) {
  std::string input_dir = "cache/" + cache_id + ".txt";
  std::ifstream ifs(input_dir);
  if (!ifs.is_open()) {
    return;
  }

  Eigen::VectorXd vec;
  int size;
  ifs >> size;
  vec.resize(size);
  for (int i = 0; i < size; i++) {
    ifs >> vec(i);
  }

  dst = vec;
}

// Eigen::MatrixXdをcacheファイルに書き込む
void save_matrix_cache(const Eigen::MatrixXd data, std::string cache_id) {
  std::string output_dir = "cache/" + cache_id + ".txt";
  std::ofstream ofs(output_dir);

  ofs << data.rows() << " " << data.cols() << "\n";

  for (int i = 0; i < data.rows(); i++) {
    for (int j = 0; j < data.cols(); j++) {
      char buffer[50];
      sprintf(buffer, "%.12lf", data(i, j));
      std::string data_str = buffer;
      ofs << data_str << "\n";
    }
  }
}

// cacheファイルからEigen::MatrixXdを読み込む
void load_matrix_cache(Eigen::MatrixXd &dst, std::string cache_id) {
  std::string input_dir = "cache/" + cache_id + ".txt";
  std::ifstream ifs(input_dir);
  if (!ifs.is_open()) {
    return;
  }

  Eigen::MatrixXd mat;
  int rows, cols;
  ifs >> rows >> cols;
  mat.resize(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      ifs >> mat(i, j);
    }
  }

  dst = mat;
}
