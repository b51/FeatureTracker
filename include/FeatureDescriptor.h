/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: FeatureDescriptor.h
 *
 *          Created On: Sat 29 Jun 2019 06:04:12 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#ifndef BINARY_FEATURE_STORE_H_
#define BINARY_FEATURE_STORE_H_

#include <Eigen/Core>
#include <algorithm>
#include <random>
#include <vector>

#include <glog/logging.h>

static const int kDefaultReservedSpace = 2000;

class FeatureDescriptor {
 public:
  FeatureDescriptor();

  FeatureDescriptor(int descriptor_size,
                     int reserved_space = kDefaultReservedSpace);

  // Set the number of bytes per descriptor. Any existing data is invalidated.
  // The number of features is always zero following this call; the
  // reserved_space argument only reserves space so that future resize() calls
  // avoid reallocation.
  void Configure(int descriptor_size,
                 int reserved_space = kDefaultReservedSpace);

  // Change the number of features. If n > num_features_ then this function
  // may trigger a re-allocation. If n < num_features_ then this function
  // never re-allocates. In either case the previous data is always retained.
  void Resize(int n);

  void Clear();

  inline uint32_t Size() const { return num_features_; }

  inline bool Empty() const { return num_features_ == 0; }

  inline uint32_t DescriptorSize() const { return descriptor_size_; }

  inline unsigned char* descriptor(uint32_t index) {
    CHECK_LT(index, num_features_);
    return &descriptor_data_[index * descriptor_size_];
  }

  inline const unsigned char* descriptor(uint32_t index) const {
    CHECK_LT(index, num_features_);
    return &descriptor_data_[index * descriptor_size_];
  }

  inline void SetDescriptorToZero(uint32_t index) {
    CHECK_LT(index, num_features_);
    memset(descriptor(index), 0, descriptor_size_);
  }

  inline void SetDescriptorRandom(uint32_t index, int seed) {
    CHECK_LT(index, num_features_);
    std::mt19937 generator(seed);
    unsigned char* data = descriptor(index);
    for (size_t i = 0; i < descriptor_size_; ++i) {
      data[i] = generator() % 256;
    }
  }

  inline unsigned char* descriptor_data() { return descriptor_data_.data(); }
  inline const unsigned char* descriptor_data() const {
    return descriptor_data_.data();
  }

  void Swap(FeatureDescriptor* other);

  // Copy the feature store from another. The preferred method of moving data is
  // with swap.
  void Copy(const FeatureDescriptor& other);

 private:
  // Prevent accidental use of the copy constructor since it is expensive.
  // The preferred way to move feature stores around is with swap().
  FeatureDescriptor(const FeatureDescriptor&) = delete;
  FeatureDescriptor& operator=(const FeatureDescriptor&) = delete;

  // The number of features in the store.
  uint32_t num_features_;

  // Number of bytes per descriptor.
  uint32_t descriptor_size_;

  std::vector<unsigned char> descriptor_data_;
};

#endif
