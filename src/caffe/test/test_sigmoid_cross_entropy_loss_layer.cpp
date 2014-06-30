#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SigmoidCrossEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SigmoidCrossEntropyLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_targets_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    data_filler.Fill(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    // Fill the targets vector
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(-1);
    targets_filler_param.set_max(1);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    targets_filler.Fill(blob_bottom_targets_);
    int count = blob_bottom_targets_->count();
    caffe_cpu_sign(count, this->blob_bottom_targets_->cpu_data(),
      this->blob_bottom_targets_->mutable_cpu_data());
    blob_bottom_vec_.push_back(blob_bottom_targets_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SigmoidCrossEntropyLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
    delete blob_top_loss_;
  }

  Dtype SigmoidCrossEntropyLossReference(const int count, const int num,
                                         const Dtype* input,
                                         const Dtype* target) {
    Dtype loss = 0;
    for (int i = 0; i < count; ++i) {
      const Dtype prediction = 1 / (1 + exp(-input[i]));
      EXPECT_LE(prediction, 1);
      EXPECT_GE(prediction, 0);
      EXPECT_LE(target[i], 1);
      EXPECT_GE(target[i], -1);
      if (target[i] != 0) {
        loss -= (target[i] > 0) * log(prediction + (target[i] < 0));
        loss -= (target[i] < 0) * log(1 - prediction + (target[i] > 0));
      }
    }
    return loss / num;
  }

  void TestForward() {
    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(-1);
    targets_filler_param.set_max(1);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    const int count = this->blob_bottom_data_->count();
    Dtype eps = 2e-2;
    for (int i = 0; i < 10; ++i) {
      // Fill the data vector
      data_filler.Fill(this->blob_bottom_data_);
      // Fill the targets vector
      targets_filler.Fill(this->blob_bottom_targets_);
      // Make negatives into -1 and positives into 1
      Dtype* targets = this->blob_bottom_targets_->mutable_cpu_data();
      caffe_cpu_sign(count, targets, targets);
      SigmoidCrossEntropyLossLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      Dtype layer_loss =
          layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      const int count = this->blob_bottom_data_->count();
      const int num = this->blob_bottom_data_->num();
      const Dtype* blob_bottom_data = this->blob_bottom_data_->cpu_data();
      const Dtype* blob_bottom_targets =
          this->blob_bottom_targets_->cpu_data();
      Dtype reference_loss = kLossWeight * SigmoidCrossEntropyLossReference(
          count, num, blob_bottom_data, blob_bottom_targets);
      EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
    }
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

<<<<<<< HEAD
TYPED_TEST_CASE(SigmoidCrossEntropyLossLayerTest, TestDtypesAndDevices);
=======
typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(SigmoidCrossEntropyLossLayerTest, Dtypes);

TYPED_TEST(SigmoidCrossEntropyLossLayerTest, TestSetup1Top) {
  LayerParameter layer_param;
  SigmoidCrossEntropyLossLayer<TypeParam> layer(layer_param);
  vector<Blob<TypeParam>*> aux_top_vec;
  Blob<TypeParam>* blob_top_ = new Blob<TypeParam>();
  aux_top_vec.push_back(blob_top_);
  layer.SetUp(this->blob_bottom_vec_, &(aux_top_vec));
  EXPECT_EQ(blob_top_->num(), 1);
  EXPECT_EQ(blob_top_->channels(), 1);
  EXPECT_EQ(blob_top_->height(), 1);
  EXPECT_EQ(blob_top_->width(), 1);
}

TYPED_TEST(SigmoidCrossEntropyLossLayerTest, TestSetup2Tops) {
  LayerParameter layer_param;
  SigmoidCrossEntropyLossLayer<TypeParam> layer(layer_param);
  vector<Blob<TypeParam>*> aux_top_vec;
  Blob<TypeParam>* blob_top_ = new Blob<TypeParam>();
  Blob<TypeParam>* blob_top2_ = new Blob<TypeParam>();
  aux_top_vec.push_back(blob_top_);
  aux_top_vec.push_back(blob_top2_);
  layer.SetUp(this->blob_bottom_vec_, &(aux_top_vec));
  EXPECT_EQ(blob_top_->num(), 1);
  EXPECT_EQ(blob_top_->channels(), 1);
  EXPECT_EQ(blob_top_->height(), 1);
  EXPECT_EQ(blob_top_->width(), 1);
  EXPECT_EQ(blob_top2_->num(), this->blob_bottom_targets_->num());
  EXPECT_EQ(blob_top2_->channels(), this->blob_bottom_targets_->channels());
  EXPECT_EQ(blob_top2_->height(), this->blob_bottom_targets_->height());
  EXPECT_EQ(blob_top2_->width(), this->blob_bottom_targets_->width());
}
>>>>>>> 889021b... Adapt test_sigmoid_sross_entropy to handle -1,0,1 labels

TYPED_TEST(SigmoidCrossEntropyLossLayerTest, TestSigmoidCrossEntropyLoss) {
  this->TestForward();
}

TYPED_TEST(SigmoidCrossEntropyLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  SigmoidCrossEntropyLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}


}  // namespace caffe
