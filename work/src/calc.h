#include <eigen3/Eigen/Dense>

// 固有値、固有ベクトルを計算
void eigensolve_cached(const Eigen::MatrixXd &sigma, Eigen::VectorXd &eigvals,
                       Eigen::MatrixXd &eigvecs);

// 分散共分散行列を計算
Eigen::MatrixXd
calc_covariance_cached(const std::vector<std::vector<Eigen::VectorXd>> &data);
