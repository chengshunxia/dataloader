#include <assert.h> 
#include <signal.h>
#include <vector>
#include <random>
#include <iostream>
#include "transforms.hpp"

using namespace cv;

Transforms::Transforms (bool enableRandomHFlip,
  bool enableRandomSizeCrop,
  bool enabelNormalize,
  int channel,
  int  dstImageHeight,
  int  dstImageWidth,

  float randomFlipProb) {
  
  this->enableRandomHFlip = enableRandomHFlip;
  this->enableRandomSizeCrop = enableRandomSizeCrop;
  this->enabelNormalize = enabelNormalize;
  this->channel = channel;
  this->dstImageHeight = dstImageHeight;
  this->dstImageWidth = dstImageWidth;
  this->randomFlipProb = randomFlipProb;
}

cv::Mat Transforms::transform(cv::Mat& input) {
  cv::Mat ret;

  //do resize/RandomCropResize
  cv::Mat resized;
  if (!this->enableRandomSizeCrop) {
      resized = this->resizeMat(input);
  } else {
    resized = randomResizedCrop(input);
  }
  
  //Flip
  cv::Mat flipped;
  if (enableRandomHFlip) {
    flipped = randomHFlip(resized);
  } else {
    flipped = resized.clone();
  }

  //Normalize or Convert to FP32 directly
  cv::Mat fp32;
  if (!enabelNormalize) {
    flipped.convertTo(fp32,CV_32FC3, 1.0 / 255.0);
  } else {
    //call normalize function
    fp32 = normalize(flipped);
  }

  ret = fp32.clone();

  fp32.release();
  flipped.release();
  resized.release();
  
  return ret;
}

cv::Mat Transforms::randomResizedCrop(cv::Mat& input) {

  //don't crop temporarily 
  Mat output;
  Size dstSize(this->dstImageHeight,
               this->dstImageWidth);
  cv::resize(input,output,dstSize);
  return output.clone();
}

cv::Mat Transforms::resizeMat(cv::Mat& input){
  Mat output;
  Size dstSize(this->dstImageHeight,
               this->dstImageWidth);
  cv::resize(input,output,dstSize);
  return output.clone();
}

cv::Mat Transforms::randomHFlip(cv::Mat& input) {
  cv::Mat output;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  if (dis(gen) > this->randomFlipProb) {
    cv::flip(input,output,1);
    if (!output.data) {
      raise(SIGSEGV);
    }
    return output.clone();
  }
  return input.clone();
}

cv::Mat Transforms::normalize(cv::Mat& input) {
  cv::Mat output;

  return output.clone();
}
