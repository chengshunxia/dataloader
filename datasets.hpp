// Copyright (c) 2018 Graphcore Ltd. All rights reserved.                                                                                                                
#ifndef _DATASETS_HPP
#define _DATASETS_HPP

#include <map>
#include <vector>
#include <iostream>

using namespace std;

class ImagenetDatasets{
public:
  ImagenetDatasets(string imagefolder,bool isTraining);
  vector<pair<string,int>> get_all_images();
  int len();
  static std::string getFileExtension(std::string filePath);

private:
  void make_dataset();
  string imageFolderPath;
  string trainImagePath;
  string validationImagePath;
  map<string,int> classIdx;
  vector<pair<string,int>> images;
  int totalImages;
  bool isTraining;
};
#endif
