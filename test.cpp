#include <set>
#include <iostream>
#include "datasets.hpp"

#define ECHO 
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
  #ifdef ECHO
  for (vector<pair<string,int>>::iterator itr = images.begin(); itr != images.end(); itr++ ) {
    cout << itr->first << " " << itr->second << endl;
  }
  #endif
}
