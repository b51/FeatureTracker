/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: FeatureDescriptor.cc
 *
 *          Created On: Thu 04 Jul 2019 04:33:41 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#include "FeatureDescriptor.h"
#include <algorithm>

FeatureDescriptor::FeatureDescriptor()
    : num_features_(0), descriptor_size_(0) {}

FeatureDescriptor::FeatureDescriptor(int descriptorSize, int reservedSpace)
    : num_features_(0), descriptor_size_(descriptorSize) {
  descriptor_data_.reserve(reservedSpace * descriptor_size_);
}

void FeatureDescriptor::Configure(int descriptorSize, int reservedSpace) {
  descriptor_size_ = descriptorSize;
  Resize(0);
  descriptor_data_.reserve(reservedSpace * descriptor_size_);
}

void FeatureDescriptor::Resize(int n) {
  num_features_ = n;
  descriptor_data_.resize(n * descriptor_size_);
}

void FeatureDescriptor::Clear() {
  num_features_ = 0;
  descriptor_data_.clear();
}

void FeatureDescriptor::Swap(FeatureDescriptor* other) {
  CHECK_NE(other, this);
  CHECK_NOTNULL(other);
  std::swap(num_features_, other->num_features_);
  std::swap(descriptor_size_, other->descriptor_size_);
  std::swap(descriptor_data_, other->descriptor_data_);
}

void FeatureDescriptor::Copy(const FeatureDescriptor& other) {
  num_features_ = other.num_features_;
  descriptor_size_ = other.descriptor_size_;
  descriptor_data_ = other.descriptor_data_;
}
