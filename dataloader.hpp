#ifndef _DATALOADER_HPP
#define _DATALOADER_HPP

#include <vector>
#include <utility>
#include <memory>
#include <boost/thread/thread.hpp> 
#include <boost/thread/mutex.hpp>
#include <boost/thread/latch.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include "datasets.hpp"
#include "transforms.hpp"
#include "blocking_queue.hpp"

using namespace cv;
using namespace std;
namespace py = pybind11;

struct fileReadRequest {
  std::string imgFilePath;
  float *arr; 
  int index;
  boost::latch* gen_latch;
};

class MemChunk {
  public:
    float * images;
    int * labels;

    MemChunk(int samplesPerStep, int channel, int height, int width) {
      images = (float*) PyMem_Malloc(sizeof (float*) * samplesPerStep * channel * height * width);
      labels = (int*) PyMem_Malloc(sizeof (int*) * samplesPerStep);
      assert(images);
      assert(labels);
   }
   ~MemChunk() {
    if (images) {
       PyMem_Free(images);
     }
    if (labels) {
       PyMem_Free(labels);
     }
     std::cout << "MemChunk freed" << std::endl;
   }
};

struct Batch {
  public:
  MemChunk *mem;
  py::array images;
  py::array labels;
  Batch(py::array images ,py::array labels, MemChunk *mem)
    : images(images),
      labels(labels),
      mem(mem){
  }
  ~Batch(){
    if (this->mem->images) {
      free(this->mem->images);
    }
    if (this->mem->labels) {
      free(this->mem->labels);
    }
    free(this->mem);

    std::cout <<"memory chunk freed" << std::endl;
   }
  py::array image() {
    return this->images;
  } 
  
  py::array label() {
    return this->labels;
  } 
};

class Dataloader {
public:
  Dataloader(ImagenetDatasets ds, 
    Transforms transform,
    int epochs,
    int samplesPerStep,
    int numWorkers,
    float prefetchNum,
    bool isTraining,
    bool drop = true,
    bool shuffle = true);
    
  Batch* next();
  ~Dataloader();
  vector<pair<string,int>> get_next_batch_images_info();
  int len();
  int get_steps_per_epoch();
  
private:
  boost::thread_group* workers;
  boost::thread* masterThread;
  ImagenetDatasets ds;
  int epochs;
  int samplesPerStep;
  int numWorkers;
  float prefetchNum;
  bool isTraining;
  bool drop;
  bool shuffle;
  int channel;
  int height;
  int width;
  int stepsPerEpoch;
  int totalSteps;
  int curtTotalSteps;
  int curStepsPerEpoch;
  int curEpoch;
  int totalSendSteps;
  BoundedBlockingQueue<Batch *>* batches;
  //boost::mutex fetchMutex;
  BoundedBlockingQueue<fileReadRequest*>* imgReadQueue;
  Transforms transforms;
  set<int> remainedImagesIdx;
  vector<pair<string,int>> originalImagesInfo;

  void work_thread();
  void master_thread();
  void get_all_samples_Idx();
};

#endif