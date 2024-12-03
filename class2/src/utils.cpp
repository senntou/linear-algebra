#include "utils.h"
#include <map>
#include <string>
#include <sys/stat.h>

void myImWrite(std::string filename, const cv::Mat &img) {
  static std::map<std::string, int> counter;
  std::string output_dir = "output/" + filename + "_";
  output_dir += std::to_string(++counter[filename]) + ".png";

  cv::imwrite(output_dir, img);
}
