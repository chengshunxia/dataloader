#ifndef _TRANSFORMS_HPP
#define _TRANSFORMS_HPP
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

class Transforms {
public:
  Transforms(bool enableRandomHFlip,
            bool enableRandomSizeCrop,
            bool enabelNormalize,
            int channel,
            int  dstImageHeight,
            int  dstImageWidth,
            float randomFlipProb = 0.5);
  cv::Mat transform(cv::Mat& input);
  cv::Mat resizeMat(cv::Mat& input);
  cv::Mat randomResizedCrop(cv::Mat& input);
  cv::Mat randomHFlip(cv::Mat& input);
  cv::Mat normalize(cv::Mat& input);

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