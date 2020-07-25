#include <assert.h>
#include <algorithm>
#include <iostream>
#include <boost/thread/thread.hpp> 
#include <boost/thread/latch.hpp>
#include <boost/python/numpy.hpp>
#include <boost/random/random_number_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "transforms.hpp"

using namespace cv;
namespace py = boost::python;
namespace np = boost::python::numpy;


cv::Mat getImaage(string file){
  Mat image;
  image = imread(file, CV_LOAD_IMAGE_COLOR);  
  if(image.empty())
  {
    std::cout <<  "Could not open or find the image " << file << std::endl;
  }
  return image.clone();
}


int main(int argc, char ** argv) {
  if (argc != 2) {
    cout << argv[0] << " Usage : " << endl;
    cout << "\t" << argv[0] << " imagePath" << endl;
    exit(1);
  }
  string path(argv[1]);
  Mat image = getImaage(path);

  std::cout << "Mat" << std::endl;
  std::cout << image << std::endl;
  image.release();
  image.release();

  return 0;
}
