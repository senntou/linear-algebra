#pragma once
#include <eigen3/Eigen/Dense>
#include <opencv2/highgui.hpp>

const int DIM = 32;
const int N = 100; // rのstep数
const int M = 100; // thetaのstep数

#define OUTPUT_DIR "output/"

// 回転した直線の画像を生成
cv::Mat get_rotation_matrix(double r, double theta);

// 画像の保存
void myImWrite(std::string filename, const cv::Mat &img);

// matrixをCSVに保存
void save_matrix_to_csv(const std::string &filename,
                        const Eigen::MatrixXd &matrix);

// ヒートマップの表示
// max_255=trueの場合、そのままの値で表示される
// max_255=falseの場合、最大値が255になるように正規化される
void show_heatmap(const Eigen::MatrixXd &data, std::string filename,
                  bool max_255 = true);
void show_heatmap(const Eigen::VectorXd &data, std::string filename,
                  bool max_255 = true);

// 固有値、固有ベクトルを計算
void eigensolve_cached(const Eigen::MatrixXd &sigma, Eigen::VectorXd &eigvals,
                       Eigen::MatrixXd &eigvecs, std::string cache_id);
