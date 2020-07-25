#include <set>
#include <iostream>
#include "datasets.hpp"
#include <boost/python.hpp>

#include "transforms.hpp"
#include "dataloader.hpp"

#define PRINT_SHUFFLE_BATCH_INDICE

using namespace std;
int main(int argc, char ** argv) {
  if (argc != 2) {
    cout << argv[0] << " Usage : " << endl;
    cout << "\t" << argv[0] << " imagenetDir" << endl;
    exit(1);
  }
  bool training = true;
  ImagenetDatasets ds(argv[1], training);


  auto images = ds.get_all_images();
  #ifdef ECHO_IMAGE
  for (vector<pair<string,int>>::iterator itr = images.begin(); itr != images.end(); itr++ ) {
    cout << itr->first << " " << itr->second << endl;
  }
  #endif

  training = false;
  ImagenetDatasets ds_val(argv[1], training);

  images = ds_val.get_all_images();
  #ifdef ECHO_IMAGE
  for (vector<pair<string,int>>::iterator itr = images.begin(); itr != images.end(); itr++ ) {
    cout << itr->first << " " << itr->second << endl;
  }
  #endif

  py::list stds;
  py::list means;
  stds.append(0.229);
  stds.append(0.224);
  stds.append(0.225);

  means.append(0.485);
  means.append(0.456);
  means.append(0.406);

  Transforms t(false,false,false,3,224,224,stds,means);
  Dataloader dl(ds, 
                  t,
                  150,
                  4,
                  8,
                  16,
                  8,
                  48,
                  100,
                  true,
                  true,
                  false);

 vector<pair<string,int>> batch = dl.get_next_batch_images_info();
 
 #ifdef PRINT_NOSHUFFLE_BATCH_INDICE
 for (auto i = 0; i < batch.size(); i++) {
   std::cout << batch[i].first << " " << batch[i].second << std::endl;
 }
 #endif
  bool shuffle = true;
  Dataloader dl_shuffle(ds, 
                  t,
                  150,
                  4,
                  8,
                  16,
                  8,
                  48,
                  100,
                  true,
                  true,
                  shuffle);

 #ifdef PRINT_SHUFFLE_BATCH_INDICE
 batch = dl_shuffle.get_next_batch_images_info();
 for (auto i = 0; i < batch.size(); i++) {
   std::cout << batch[i].first << " " << batch[i].second << std::endl;
 }
 #endif




}
