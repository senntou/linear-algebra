#include "serialize.h"
#include <string>

string to_string_accurate(double x) {
  char buf[32];
  sprintf(buf, "%.12f", x);
  return string(buf);
}

// keyの生成
string generate_key_by_Matrix(MatrixXd input) {
  string key = "";
  for (int i = 0; i < input.rows(); i++) {
    for (int j = 0; j < input.cols(); j++) {
      key += to_string(input(i, j)) + ",";
    }
  }
  return key;
}

// 固有値・固有ベクトルのシリアライズ
string serialize_eigenvals(pair<VectorXd, MatrixXd> output) {
  // まず固有値
  int length = output.first.size();
  string str = to_string(length) + ",";
  for (int i = 0; i < length; i++) {
    str += to_string_accurate(output.first(i)) + ",";
  }

  // 次に固有ベクトル
  int rows = output.second.rows();
  int cols = output.second.cols();
  str += to_string(rows) + "," + to_string(cols) + ",";
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      str += to_string_accurate(output.second(i, j)) + ",";
    }
  }

  return str;
}

// 固有値・固有ベクトルのデシリアライズ
pair<VectorXd, MatrixXd> deserialize_eigenvals(string str) {
  vector<double> values;
  string value = "";
  for (int i = 0; i < str.size(); i++) {
    if (str[i] == ',') {
      values.push_back(stod(value));
      value = "";
    } else {
      value += str[i];
    }
  }

  int length = round(values[0]);
  VectorXd eigvals = VectorXd::Zero(length);
  for (int i = 0; i < length; i++) {
    eigvals(i) = values[i + 1];
  }

  int rows = round(values[length + 1]);
  int cols = round(values[length + 2]);

  assert("serialize.cpp:84" && values.size() == length + 3 + rows * cols);
  MatrixXd eigvecs = MatrixXd::Zero(rows, cols);
  int index = length + 3;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      eigvecs(i, j) = values[index];
      index++;
    }
  }

  return make_pair(eigvals, eigvecs);
}

// matrixのシリアライズ
string serialize_matrix(MatrixXd input) {
  int rows = input.rows();
  int cols = input.cols();
  string str = to_string(rows) + "," + to_string(cols) + ",";
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      str += to_string_accurate(input(i, j)) + ",";
    }
  }
  return str;
}

// matrixのデシリアライズ
MatrixXd deserialize_matrix(string str) {
  vector<double> values;
  string value = "";
  for (int i = 0; i < str.size(); i++) {
    if (str[i] == ',') {
      values.push_back(stod(value));
      value = "";
    } else {
      value += str[i];
    }
  }

  int rows = round(values[0]);
  int cols = round(values[1]);

  assert("serialize.cpp:128" && values.size() == 2 + rows * cols);
  MatrixXd matrix = MatrixXd::Zero(rows, cols);
  int index = 2;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix(i, j) = values[index];
      index++;
    }
  }

  return matrix;
}
