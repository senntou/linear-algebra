#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <opencv2/highgui.hpp>

template <typename T, int N>
void print_vector(std::string title, const Eigen::Vector<T, N> &v) {
  std::cout << "== Vector ==" << std::endl;
  std::cout << title << std::endl;
  std::cout << "size: " << v.size() << std::endl;
  std::cout << v << std::endl;
}

template <typename T, int N, int M>
void print_matrix(std::string title, const Eigen::Matrix<T, N, M> &m) {
  std::cout << "== Matrix ==" << std::endl;
  std::cout << title << std::endl;
  std::cout << "size: " << m.rows() << "x" << m.cols() << std::endl;
  std::cout << m << std::endl;
}

template <typename T, int N>
void vector_right_shift(Eigen::Vector<T, N> &v, int shift) {
  Eigen::Vector<T, N> tmp = v;
  for (int i = 0; i < N; i++) {
    tmp(i) = v((i - shift + N) % N);
  }
  v = tmp;
}

void myImWrite(std::string filename, const cv::Mat &img);
