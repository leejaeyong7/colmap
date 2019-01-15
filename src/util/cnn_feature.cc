#include "util/cnn_feature.h"

using namespace colmap;

CNNFeature::CNNFeature(){

}

CNNFeature::CNNFeature(const CNNFeature& other){
  data_ = other.data_;
  width_ = other.width_;
  height_ = other.height_;
  channels_ = other.channels_; 
}

CNNFeature::CNNFeature(cnpy::NpyArray * data){
  data_ = data;
  auto shape = data->shape;
  width_ = shape[0];
  height_ = shape[1];
  channels_ = shape[2]; 
}

CNNFeature& CNNFeature::operator=(const CNNFeature& other){
  data_ = other.data_;
  width_ = other.width_;
  height_ = other.height_;
  channels_ = other.channels_;
  return *this;
}
CNNFeature::~CNNFeature(){
  if(data_ != nullptr){
    delete data_;
  }
}

std::vector<float> CNNFeature::ConvertToRowMajorArray()const{
  auto v = std::vector<float>();
  v.assign(Data(), Data() + NumBytes());
  return v;
}

void CNNFeature::Rescale(const int new_width, const int new_height){

}

/**
 * Reads NPY file and assign it to data variable
 * @param  path full path to feature file
 * @return      true on success, else false
 */
bool CNNFeature::Read(const std::string& path){
  try{
    data_ = new cnpy::NpyArray()
    *data_ = cnpy::npy_load(path);
    auto shape = data_.shape;
    width_ = shape[0];
    height_ = shape[1];
    channels_ = shape[2];
    return true;
  } catch(std::exception& e){
    return false;
  }
}

bool CNNFeature::Write(const std::string& path) const{
  try{
    auto shape = data_->shape;
    auto data = data_->data<float>();
    cnpy::npy_save(path, data, shape);
    return true;
  } catch(std::exception& e){
    return false;
  }
}


float * CNNFeature::Data() { return data_->data<float>(); }
const float * CNNFeature::Data() const { return data_->data<float>(); }

int CNNFeature::Width() const { return width_; }
int CNNFeature::Height() const { return height_; }
int CNNFeature::Channels() const { return channels_; }
size_t CNNFeature::NumBytes() const{
  return data_.word_size;
};
