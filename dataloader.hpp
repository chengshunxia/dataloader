#ifndef _DATALOADER_HPP
#define _DATALOADER_HPP

#include <vector>
#include <utility>
#include <boost/python.hpp>
#include <boost/thread/thread.hpp> 
#include <boost/thread/mutex.hpp>
#include <boost/thread/latch.hpp>

#include "datasets.hpp"
#include "transforms.hpp"
#include "blocking_queue.hpp"
#include "transforms.hpp"

using namespace cv;
namespace py = boost::python;

struct fileReadRequest {
  std::string imgFilePath;
  float *arr; 
  int index;
  boost::latch* gen_latch;
};

class Dataloader {
public:
  Dataloader(ImagenetDatasets ds, 
    Transforms transform,
    int epochs,
    int batchPerGraph,
    int replicationFactor,
    int gradientAcclFactor,
    int batchPerStep,
    int numWorkers,
    float prefetchNum,
    bool isTraining,
    bool drop = true,
    bool shuffle = true);
    
  py::tuple next();
  void batchRelease(py::tuple tp);
  ~Dataloader();
  vector<pair<string,int>> get_next_batch_images_info();
  int len();
  

private:
  boost::thread_group workers;
  boost::thread masterThread;
  ImagenetDatasets ds;
  int epochs;
  int batchPerGraph;
  int replicationFactor;
  int gradientAcclFactor;
  int batchPerStep;
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
  BoundedBlockingQueue<py::tuple> batches;
  boost::mutex fetchMutex;
  BoundedBlockingQueue<fileReadRequest*> imgReadQueue;
  Transforms transforms;
  set<int> remainedImagesIdx;
  vector<pair<string,int>> originalImagesInfo;

  void work_thread();
  void master_thread();
  void get_all_samples_Idx();
};

#endif