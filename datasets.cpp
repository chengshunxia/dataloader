#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "datasets.hpp"

using namespace std;
namespace fs = boost::filesystem;

ImagenetDatasets::ImagenetDatasets(string imagefolder, bool isTraining) {
  fs::path imagesRootDir(imagefolder);
  if (!fs::exists(imagesRootDir) || 
      !fs::is_directory(imagesRootDir)) {
    string errMsg = imagefolder + "does not exists or is not a directory";
    /*
    throw fs::filesystem_error(errMsg,
                               boost::system::errc::no_such_file_or_directory);
    */
  }
  this->imageFolderPath = imagefolder;
  this->trainImagePath = imagefolder + "/train";
  this->validationImagePath = imagefolder + "/val";
  if (isTraining) {
      fs::path trainImagesDir(this->imageFolderPath);
      if (!fs::exists(trainImagesDir) || 
          !fs::is_directory(trainImagesDir)) {
        string errMsg = this->imageFolderPath + "does not exists or is not a directory";
        /*
        throw fs::filesystem_error(errMsg,
                                   boost::system::errc::no_such_file_or_directory);
        */
      }
  } else {
    fs::path validationImagesDir(this->validationImagePath);
    if (!fs::exists(validationImagesDir) || 
          !fs::is_directory(validationImagesDir)) {
      string errMsg = this->validationImagePath  + "does not exists or is not a directory";
      /*
      throw fs::filesystem_error(errMsg,
                                 boost::system::errc::no_such_file_or_directory);
      */
    }
  }
  make_dataset();
}

std::string ImagenetDatasets::getFileExtension(std::string filePath)
{
  fs::path pathObj(filePath);
  if (pathObj.has_extension()) {
    return pathObj.extension().string();
  }
  return "";
}

void ImagenetDatasets::make_dataset() {
  string _dir = this->isTraining ? this->trainImagePath : this->validationImagePath;
  fs::directory_iterator end_iter;
  vector<string> dirs;
  for (fs::directory_iterator dir_itr(_dir);
       dir_itr != end_iter;
       ++dir_itr ) {
    if (!fs::is_directory(dir_itr->status())) {
        continue;
    }
    dirs.push_back(dir_itr->path().string());
  }
  sort(dirs.begin(),dirs.end());

  int file_count = 0;
  for (auto i = 0; i < dirs.size(); i++) {
    classIdx[dirs[i]] = i;
    fs::directory_iterator sub_end_iter;
    for (fs::directory_iterator subdir_itr(dirs[i]);
         subdir_itr != sub_end_iter;
         ++subdir_itr) {
      if (fs::is_regular_file( subdir_itr->status())) {
        string filename = subdir_itr->path().string();
        auto extension = getFileExtension(filename);
        if (extension == ".jpeg" 
            || extension == ".JPEG") {
          this->images.push_back(std::make_pair(filename,i));
          file_count++;
        }
      }
    }
  }
  this->totalImages = file_count;
}

vector<pair<string,int>> ImagenetDatasets::get_all_images(){
  return this->images;
}

int ImagenetDatasets::len(){
  return this->totalImages;
}