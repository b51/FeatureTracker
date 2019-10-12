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

static const int kDefaultReservedSpace = 2000;

template <typename ScalarType>
class FeatureDescriptor {
 public:
  FeatureDescriptor() : num_features_(0), descriptor_size_(0) {}

  FeatureDescriptor(int descriptor_size,
                    int reserved_space = kDefaultReservedSpace)
      : num_features_(0), descriptor_size_(descriptor_size) {
    descriptor_data_.reserve(reserved_space * descriptor_size_);
  }

  // Set the number of bytes per descriptor. Any existing data is invalidated.
  // The number of features is always zero following this call; the
  // reserved_space argument only reserves space so that future resize() calls
  // avoid reallocation.
  void Configure(int descriptor_size,
                 int reserved_space = kDefaultReservedSpace) {
    descriptor_size_ = descriptor_size;
    Resize(0);
    descriptor_data_.reserve(reserved_space * descriptor_size_);
  }

  // Change the number of features. If n > num_features_ then this function
  // may trigger a re-allocation. If n < num_features_ then this function
  // never re-allocates. In either case the previous data is always retained.
  void Resize(int n) {
    num_features_ = n;
    descriptor_data_.resize(n * descriptor_size_);
  }

  void Clear() {
    num_features_ = 0;
    descriptor_data_.clear();
  }

  inline uint32_t Size() const { return num_features_; }

  inline bool Empty() const { return num_features_ == 0; }

  inline uint32_t DescriptorSize() const { return descriptor_size_; }

  inline ScalarType* descriptor(uint32_t index) {
    // CHECK_LT(index, num_features_);
    return &descriptor_data_[index * descriptor_size_];
  }

  inline const ScalarType* descriptor(uint32_t index) const {
    // CHECK_LT(index, num_features_);
    return &descriptor_data_[index * descriptor_size_];
  }

  inline void SetDescriptorToZero(uint32_t index) {
    // CHECK_LT(index, num_features_);
    memset(descriptor(index), 0, descriptor_size_);
  }

  inline void SetDescriptorRandom(uint32_t index, int seed) {
    // CHECK_LT(index, num_features_);
    std::mt19937 generator(seed);
    ScalarType* data = descriptor(index);
    for (size_t i = 0; i < descriptor_size_; ++i) {
      data[i] = generator() % 256;
    }
  }

  inline ScalarType* descriptor_data() { return descriptor_data_.data(); }
  inline const ScalarType* descriptor_data() const {
    return descriptor_data_.data();
  }

  void Swap(FeatureDescriptor* other) {
    CHECK_NE(other, this);
    CHECK_NOTNULL(other);
    std::swap(num_features_, other->num_features_);
    std::swap(descriptor_size_, other->descriptor_size_);
    std::swap(descriptor_data_, other->descriptor_data_);
  }

  // Copy the feature store from another. The preferred method of moving data is
  // with swap.
  void Copy(const FeatureDescriptor& other) {
    num_features_ = other.num_features_;
    descriptor_size_ = other.descriptor_size_;
    descriptor_data_ = other.descriptor_data_;
  }

 private:
  // Prevent accidental use of the copy constructor since it is expensive.
  // The preferred way to move feature stores around is with swap().
  FeatureDescriptor(const FeatureDescriptor&) = delete;
  FeatureDescriptor& operator=(const FeatureDescriptor&) = delete;

  // The number of features in the store.
  uint32_t num_features_;

  // Number of bytes per descriptor.
  uint32_t descriptor_size_;

  std::vector<ScalarType> descriptor_data_;
};

using FeatureDescriptoru = FeatureDescriptor<unsigned char>;
using FeatureDescriptorf = FeatureDescriptor<float>;

#endif
