#pragma once
#include <Eigen/Dense>
#include <string>

// Eigen::VectorXdをcacheファイルに書き込む
void save_vector_cache(const Eigen::VectorXd data, std::string cache_id);

// cacheファイルからEigen::VectorXdを読み込む
void load_vector_cache(Eigen::VectorXd &dst, std::string cache_id);

// Eigen::MatrixXdをcacheファイルに書き込む
void save_matrix_cache(const Eigen::MatrixXd data, std::string cache_id);

// cacheファイルからEigen::MatrixXdを読み込む
void load_matrix_cache(Eigen::MatrixXd &dst, std::string cache_id);
