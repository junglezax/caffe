#ifndef CAFFE_MULTI_LABEL_ACCURACY_LAYER_HPP_
#define CAFFE_MULTI_LABEL_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/* MultiLabelAccuracyLayer
  Note: not an actual loss layer! Does not implement backwards step.
  Computes the accuracy of a with respect to b.
*/
template <typename Dtype>
class MultiLabelAccuracyLayer : public Layer<Dtype> {
 public:
  explicit MultiLabelAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
  }
};

}  // namespace caffe

#endif  // CAFFE_MULTI_LABEL_ACCURACY_LAYER_HPP_
