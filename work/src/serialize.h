#pragma once
#include "sqlite.h"
#include <Eigen/Dense>
#include <string>

using namespace std;
using namespace Eigen;

// keyの生成
string generate_key_by_Matrix(MatrixXd input);

// 固有値・固有ベクトルのシリアライズ
string serialize_eigenvals(pair<VectorXd, MatrixXd> output);

// 固有値・固有ベクトルのデシリアライズ
pair<VectorXd, MatrixXd> deserialize_eigenvals(string str);
