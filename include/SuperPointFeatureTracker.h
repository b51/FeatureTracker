/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: SuperPointFeatureTracker.h
 *
 *          Created On: Fri 11 Oct 2019 07:52:55 AM UTC
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#ifndef FEATURE_TRACKER_SUPER_POINT_FEATURE_TRACKER_H_
#define FEATURE_TRACKER_SUPER_POINT_FEATURE_TRACKER_H_

#include "FeatureTrackerBase.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>

#include "FeatureDescriptor.h"
#include "FeatureDetectorBase.h"
#include "SuperPointFeatureDetector.h"
#include "SuperPointFeatureMatcher.h"

class SuperPointFeatureTracker : public FeatureTrackerBase {
 public:
  SuperPointFeatureTracker();

  ~SuperPointFeatureTracker() = default;

  virtual void Init(int width, int height, int max_number_of_features);

  virtual void Track(const cv::Mat& image,
                     Eigen::Matrix2Xf* current_measurements,
                     Eigen::Matrix2Xf* previous_measurements,
                     std::vector<int>* feature_ids);

  void Init(int width, int height, int max_number_of_features,
            std::string model_path, float conf_thresh, int nms_dist,
            float nn_thresh);


  virtual bool IsInitialized() const { return is_initialized_; }

  virtual int GetWidth() const { return width_; }

  virtual int GetHeight() const { return height_; }

  virtual void GetFeatureLocations(std::vector<Eigen::Vector2d>* points) const;

  virtual const std::vector<int>& GetNewTrackIds() const {
    return previous_track_ids_;
  }

  virtual int GetNumberOfFeaturesMatched() const { return matches_.cols(); }

  void GetFeatureDescriptor(FeatureDescriptorf* descriptors) {
    descriptors->Copy(current_descriptors_);
  }

  virtual void Display();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  void LocalInit(int width, int height);

  std::shared_ptr<SuperPointFeatureDetector> detector_;
  std::shared_ptr<SuperPointFeatureMatcher> matcher_;

  int width_;
  int height_;
  int current_track_id_;
  bool is_initialized_;

  Eigen::Matrix2Xf previous_feature_locations_;
  Eigen::Matrix2Xf current_feature_locations_;
  /**
   *  matches_[0]: current_image matched keypoints index
   *          [1]: previous_image matched keypoints index
   *          [2]: distance
   */
  Eigen::Matrix3Xf matches_;

  std::vector<int> previous_track_ids_;
  std::vector<int> current_track_ids_;

  FeatureDescriptorf previous_descriptors_;
  FeatureDescriptorf current_descriptors_;

  cv::Mat previous_image_;
  cv::Mat current_image_;
};

#endif
