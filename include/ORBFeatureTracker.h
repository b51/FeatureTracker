/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: ORBFeatureTracker.h
 *
 *          Created On: Thu 26 Sep 2019 04:07:53 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#ifndef FEATURE_TRACKER_ORB_FEATURE_TRACKER_H_
#define FEATURE_TRACKER_ORB_FEATURE_TRACKER_H_

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <memory>
#include <opencv2/opencv.hpp>

#include "FeatureDescriptor.h"
#include "FeatureDetectorBase.h"
#include "ORBFeatureDetector.h"
#include "ORBFeatureMatcher.h"

class ORBFeatureTracker {
 public:
  typedef cv::NormTypes DistType;

  ORBFeatureTracker();

  ~ORBFeatureTracker() = default;

  void Init(int width, int height, int max_number_of_features);

  void Init(int width, int height, int max_number_of_features,
            float scale_factor, int number_of_levels, int edge_dist,
            int patch_size, bool refine_corners, DistType dist_type,
            float refine_dist);

  void Track(const cv::Mat& image, Eigen::Matrix2Xd* curr_measurements,
             Eigen::Matrix2Xd* prev_measurements,
             std::vector<int>* feature_ids);

  void Display();

  inline int GetWidth() const { return width_; }

  inline int GetHeight() const { return height_; }

  inline bool IsInitialized() const { return is_initialized_; }

  inline float GetScaleFactor() const { return detector_->GetScaleFactor(); }

  inline int GetEdgeDist() const { return detector_->GetEdgeDist(); }

  inline int GetPatchSize() const { return detector_->GetPatchSize(); }

  inline bool GetRefineCorners() const { return detector_->GetRefineCorners(); }

  inline DistType GetDistType() const { return matcher_->GetDistType(); }

  inline float GetRefineDist() const { return matcher_->GetRefineDist(); }

  inline int GetMaxNumberOfFeatures() const {
    return detector_->GetMaxNumberOfFeatures();
  }

  inline int GetNumberOfLevels() const {
    return detector_->GetNumberOfLevels();
  }

  void SetDetectorMask(const cv::Mat& mask);

  const std::vector<cv::DMatch>& GetMatches() const { return matches_; }

  void GetFeatureDescriptor(cv::Mat* descriptors) const {
    curr_descriptor_.copyTo(*descriptors);
  }

  void GetFeatureLocations(std::vector<Eigen::Vector2d>* points) const {
    CHECK_NOTNULL(points);
    points->resize(curr_feature_locations_.size());
    for (size_t i = 0; i < curr_feature_locations_.size(); i++) {
      (*points)[i] << curr_feature_locations_[i].pt.x,
          curr_feature_locations_[i].pt.y;
    }
  }

  const std::vector<int>& GetNewTrackIds() const { return prev_track_ids_; }

  int GetNumberOfFeaturesDetected() const {
    return curr_feature_locations_.size();
  }

  int GetNumberOfFeaturesMatched() const { return matches_.size(); }

 private:
  void LocalInit(int width, int height);

  std::unique_ptr<ORBFeatureDetector> detector_;
  std::unique_ptr<ORBFeatureMatcher> matcher_;
  int width_;
  int height_;
  int current_track_id_;
  bool is_initialized_;

  std::vector<cv::KeyPoint> prev_feature_locations_;
  std::vector<cv::KeyPoint> curr_feature_locations_;
  std::vector<cv::DMatch> matches_;
  std::vector<int> prev_track_ids_;
  std::vector<int> curr_track_ids_;

  cv::Mat prev_descriptor_;
  cv::Mat curr_descriptor_;

  // For debugging we keep the images here
  cv::Mat prev_image_;
  cv::Mat curr_image_;
};

#endif
