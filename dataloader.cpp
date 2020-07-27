#include <assert.h>
#include <algorithm>
#include <boost/thread/thread.hpp> 
#include <boost/thread/latch.hpp>
#include <boost/python/numpy.hpp>
#include <boost/random/random_number_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"


#include "datasets.hpp"
#include "transforms.hpp"
#include "blocking_queue.hpp"
#include "dataloader.hpp"

using namespace cv;
namespace py = boost::python;
namespace np = boost::python::numpy;

Dataloader::Dataloader(ImagenetDatasets ds, 
                      Transforms transforms,
                      int epochs,
                      int batchPerGraph,
                      int replicationFactor,
                      int gradientAcclFactor,
                      int batchPerStep,
                      int numWorkers,
                      float prefetchNum,
                      bool isTraining,
                      bool drop,
                      bool shuffle) 
                      : ds(ds)
                      , transforms(transforms) {
  if (isTraining) {
    assert(drop);
  }
  assert(batchPerGraph>0);
  assert(replicationFactor>0);
  assert(gradientAcclFactor>0);
  assert(batchPerStep>0);
  assert(numWorkers>=1);

  this->originalImagesInfo = ds.get_all_images();
  this->channel = transforms.channel;
  this->height = transforms.dstImageHeight;
  this->width = transforms.dstImageWidth;
  int totalSamples = ds.len();
  this->batchPerGraph = batchPerGraph;
  this->replicationFactor = replicationFactor;
  this->gradientAcclFactor = gradientAcclFactor;
  this->batchPerStep = batchPerStep;
  this->samplesPerStep = batchPerGraph * 
                         replicationFactor *
                         gradientAcclFactor * 
                         batchPerStep;

  this->stepsPerEpoch = totalSamples / this->samplesPerStep;
  this->totalSteps = this->stepsPerEpoch * epochs;
  this->curtTotalSteps = 0;
  this->curStepsPerEpoch = 0;
  this->curEpoch = 0;
  this->totalSendSteps = 0;
  this->shuffle = shuffle;

  get_all_samples_Idx();

  this->imgReadQueue = new BoundedBlockingQueue<fileReadRequest*> (1000000);
  this->batches =  new BoundedBlockingQueue<py::tuple> (prefetchNum);

  this->workers = new boost::thread_group();
  for (auto i = 0; i < numWorkers; i++ ) {
    this->workers->create_thread(boost::bind(&Dataloader::work_thread,this));
  }
  masterThread = new boost::thread(boost::bind(&Dataloader::master_thread,this));

}

//thread used to load imgs from disk
void Dataloader::work_thread(){
  while (1) {
      auto request = this->imgReadQueue->take();
      auto filename = request->imgFilePath;
      float * arr = request->arr;
      int batch_index = request->index;
      boost::latch * gen_latch = request->gen_latch;
      Mat image;
      image = imread(filename, CV_LOAD_IMAGE_COLOR);  
      if(image.empty())
      {
        std::cout <<  "Could not open or find the image " << filename << std::endl;
        gen_latch->count_down();
      }
      Mat dstImage = this->transforms.transform(image);

      std::vector<Mat> bgr_planes;
      split(dstImage,bgr_planes);

      float * _arr = arr + batch_index * this->channel * this->height * this->width;
      for (auto c = 0 ; c < this->channel ; c++) {
          int z = this->channel - 1 - c;
          float * dst_addr = _arr + z * this->height * this->width;
          for (auto j = 0; j <= this->height * this->width; j++)
            *(dst_addr+j) = static_cast<float> (bgr_planes[z].data[j]);
      }
      delete request;
      dstImage.release();
      image.release();
      gen_latch->count_down();
    }
}

void Dataloader::get_all_samples_Idx(){
  this->remainedImagesIdx.clear();
  int len = this->ds.len();
  for (auto idx = 0; idx < len; idx++) {
    this->remainedImagesIdx.insert(idx);
  }
}

vector<pair<string,int>> Dataloader::get_next_batch_images_info() {
  //reset the this->remainedImagesIdx if needed 
  if (this->remainedImagesIdx.size() < this->samplesPerStep) {
    get_all_samples_Idx();
  }
  std::vector<int> _tmp;
  _tmp.assign(this->remainedImagesIdx.begin(),this->remainedImagesIdx.end());
  vector<pair<string,int>> next_batch;
  if (this->shuffle) {
    
    boost::random::mt19937 rng;
    std::set<int> rand_indices;
    boost::random::uniform_int_distribution<int> indice(0,this->remainedImagesIdx.size() - 1);

    while (rand_indices.size() < this->samplesPerStep) {
        auto index = indice(rng);
        rand_indices.insert(index);
    }
    for (auto itr = rand_indices.begin(); itr != rand_indices.end(); itr++) {
      next_batch.push_back(this->originalImagesInfo[_tmp[*itr]]);
      this->remainedImagesIdx.erase(_tmp[*itr]);
    }
    //need to shuffle the ret vector 
    boost::random::random_number_generator<boost::mt19937> referenceRand(rng);
    std::random_shuffle(next_batch.begin(), next_batch.end(), referenceRand);
  } else {
    for (auto i = 0; i < this->samplesPerStep; i++) {
      int index = _tmp[i];
      next_batch.push_back(this->originalImagesInfo[index]);
      this->remainedImagesIdx.erase(index);
    }
  }
  return next_batch;
}

void Dataloader::master_thread(){

  while (this->totalSendSteps < this->totalSteps) {

#ifdef TPUT_LOGGING
    auto start_time = boost::posix_time::microsec_clock::universal_time();
#endif


    vector<pair<string,int>> ImagesInfo = get_next_batch_images_info();
    unsigned long long arrSize = this->samplesPerStep * 
                          this->channel *
                          this->height *
                          this->width;
    float * arr = new float[arrSize];
    int * labelArr = new int[this->samplesPerStep];
    boost::latch* gen_latch = new boost::latch(samplesPerStep);
    for (auto i = 0; i < this->samplesPerStep; i++) {
      this->imgReadQueue->put(new fileReadRequest {
        ImagesInfo[i].first,
        arr,
        this->curStepsPerEpoch,
        gen_latch
      });
      labelArr[i] = ImagesInfo[i].second;
    }

    if (! gen_latch->try_wait())
      if (gen_latch->wait_for(boost::chrono::milliseconds(100)) ==  boost::cv_status::timeout)
          if (gen_latch->wait_until(boost::chrono::steady_clock::now()+boost::chrono::milliseconds(100)) ==  boost::cv_status::timeout)
            gen_latch->wait();

    int first_dim = this->batchPerGraph * 
                    this->replicationFactor * 
                    this->gradientAcclFactor * 
                    this->batchPerStep;
    auto shape = py::make_tuple(first_dim,
                  channel,
                  height,
                  width
                );
    auto stride = py::make_tuple(this->width * this->height * this->channel,
                  this->width * this->height,
                  this->width,
                  1) ;
    np::dtype dt1 = np::dtype::get_builtin<float>();
    auto mul_data_ex = np::from_data(arr,
                    dt1,
                    shape,
                    stride,
                    py::object());
    
    auto shape_dst = py::make_tuple(this->batchPerStep,
                  this->gradientAcclFactor,
                  this->replicationFactor,
                  this->batchPerGraph,
                  this->channel,
                  this->height,
                  this->width
                );
    mul_data_ex = mul_data_ex.reshape(shape_dst);

    auto shape_label = py::make_tuple(this->batchPerStep,
                                    this->gradientAcclFactor,
                                    this->replicationFactor,
                                    this->batchPerGraph);

    stride = py::make_tuple(this->batchPerGraph * this->replicationFactor * this->gradientAcclFactor,
                          this->batchPerGraph * this->replicationFactor,
                          this->batchPerGraph,
                          1) ;

    np::dtype label_type = np::dtype::get_builtin<int>();
    auto mul_data_ex_label = np::from_data(labelArr,
                                          label_type,
                                          shape_label,
                                          stride,
                                          py::object());

    delete gen_latch;
    //format Label Numpy Array
  
    py::tuple batch = py::make_tuple(mul_data_ex, mul_data_ex_label);
    this->batches->put(batch);

#ifdef TPUT_LOGGING
    auto end_time = boost::posix_time::microsec_clock::universal_time();
    auto time_elapse = end_time - start_time;
    int ticks_cast = time_elapse.ticks();
    float avgImgsPerSecond = static_cast<float>(this->samplesPerStep) * 1000000 / static_cast<float>(ticks_cast);
    std::cout << "Time cost : " << ticks_cast <<" BPS: "<< this->samplesPerStep << " imgs/s: " << avgImgsPerSecond << std::endl;
#endif

  }
}

py::tuple Dataloader::next() {
  this->totalSendSteps++;
  this->curStepsPerEpoch++;
  return this->batches->take();
}

void Dataloader::batchRelease(py::tuple tp){
  /*
  delete[] tp;
  delete[] tp;
  */
}

int Dataloader::len() {
  return this->totalSteps;
}

int Dataloader::get_steps_per_epoch() {
  return this->stepsPerEpoch;
}

Dataloader::~Dataloader(){
  this->workers->interrupt_all();
  this->workers->join_all();
  masterThread->interrupt();
  masterThread->join();

  delete workers;
  delete masterThread;
  delete batches;
  delete imgReadQueue;
}
