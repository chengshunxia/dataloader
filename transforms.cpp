#include <assert.h> 
#include <vector>
#include <random>
#include <iostream>
#include <boost/python.hpp>

#include "transforms.hpp"

using namespace cv;
namespace py = boost::python;

Transforms::Transforms (bool enableRandomHFlip,
  bool enableRandomSizeCrop,
  bool enabelNormalize,
  int channle,
  int  dstImageHeight,
  int  dstImageWidth,
  py::list mean,
  py::list std,
  float randomFlipProb) {

  this->enableRandomHFlip = enableRandomHFlip;
  this->enableRandomSizeCrop = enableRandomSizeCrop;
  this->enabelNormalize = enabelNormalize;
  this->channel = channel;
  this->dstImageHeight = dstImageHeight;
  this->dstImageWidth = dstImageWidth;
  this->randomFlipProb = randomFlipProb;

  py::ssize_t lenMean = py::len(mean);
  py::ssize_t lenStd = py::len(std);
  assert(lenMean == lenStd);
  for (auto i = 0; i < lenMean; i++) {
    this->mean.push_back(py::extract<float>(mean[i]));
    this->std.push_back(py::extract<float>(std[i]));
  } 
}

void Transforms::transform(cv::Mat input, cv::Mat output) {

  if (!this->enableRandomSizeCrop) {
      this->resizeMat(input,output);
  } else {
    randomResizedCrop(input,output);
  }

  if (enableRandomHFlip) {

  }
}

void Transforms::randomResizedCrop(cv::Mat input, cv::Mat output) {
  cv::resize(input,output,Size(this->dstImageHeight,
                               this->dstImageWidth));
}

void Transforms::resizeMat(cv::Mat input, cv::Mat output){
  Size dstSize(this->dstImageHeight,
               this->dstImageWidth);
  cv::resize(input,output,dstSize);
}
void Transforms::randomHFlip(cv::Mat input, cv::Mat output) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  if (dis(gen) > this->randomFlipProb) {
    cv::flip(input,output,1);
  }
}

void Transforms::normalize(cv::Mat input) {
  
}

void Transforms::convertToFloat32() {

}