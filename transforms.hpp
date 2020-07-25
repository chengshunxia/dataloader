#ifndef _TRANSFORMS_HPP
#define _TRANSFORMS_HPP
#include <vector>
#include <boost/python.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
namespace py = boost::python;

class Transforms {
public:
  Transforms(bool enableRandomHFlip,
            bool enableRandomSizeCrop,
            bool enabelNormalize,
            int channel,
            int  dstImageHeight,
            int  dstImageWidth,
            py::list mean,
            py::list std,
            float randomFlipProb = 0.5);
  void transform(cv::Mat input, cv::Mat output);
  void resizeMat(cv::Mat input, cv::Mat output);
  void randomResizedCrop(cv::Mat input, cv::Mat output);
  void randomHFlip(cv::Mat input, cv::Mat output);
  void normalize(cv::Mat input);
  void convertToFloat32();
  int channel;
  int dstImageHeight;
  int dstImageWidth;

private:
  bool enableRandomHFlip;
  bool enableRandomSizeCrop;
  bool enabelNormalize;
  float randomFlipProb;
  vector<float> mean;
  vector<float> std;
};
#endif