/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: FeatureDetectorBase.h
 *
 *          Created On: Thu 27 Jun 2019 03:29:38 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#ifndef FEATURE_TRACKER_FEATURE_DETECTOR_BASE_H_
#define FEATURE_TRACKER_FEATURE_DETECTOR_BASE_H_

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "FeatureDescriptor.h"

class FeatureDetectorBase {
 public:
  FeatureDetectorBase(){};
  virtual ~FeatureDetectorBase(){};

  virtual void Init(int width, int height, int max_number_of_features) = 0;

  virtual void Detect(const cv::Mat& image,
                      Eigen::Matrix2Xd* current_measurements,
                      std::vector<float>* current_feature_orientations,
                      std::vector<float>* current_feature_scales,
                      FeatureDescriptoru* current_feature_descriptors) {}

  virtual void Detect(const cv::Mat& image,
                      Eigen::Matrix2Xf* current_measurements,
                      FeatureDescriptorf* current_feature_descriptors) {}

  virtual bool IsInitialized() const = 0;
  virtual int GetHeight() const = 0;
  virtual int GetWidth() const = 0;
  virtual int GetMaxNumberOfFeatures() const = 0;

 private:
  FeatureDetectorBase(const FeatureDetectorBase& fd) = delete;
  FeatureDetectorBase& operator=(const FeatureDetectorBase& fd) = delete;
};

#endif
