#include "serialize.h"
#include "sqlite.h"
#include "utils.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <string>

using namespace std;
using namespace Eigen;

// 固有値の計算
void eigensolve_cached(const Eigen::MatrixXd &input, Eigen::VectorXd &eigvals,
                       Eigen::MatrixXd &eigvecs) {
  static const string cache_id = "eigensolve";

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

// 分散共分散行列を計算
MatrixXd
calc_covariance_cached(const std::vector<std::vector<Eigen::VectorXd>> &data) {
  static const string cache_id = "calc_covariance";

  string input_key = to_string(N) + "_" + to_string(M);

  string values = get_cache(cache_id, input_key);

  if (values != "") {
    cout << "[CACHE HIT] " << cache_id << endl;
    return deserialize_matrix(values);
  }

  cout << "[CACHE MISS] " << cache_id << endl;

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

  // キャッシュに保存
  string serialized = serialize_matrix(sigma);
  save_cache(cache_id, input_key, serialized);

  return sigma;
}
