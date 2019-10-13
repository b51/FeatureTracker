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

#include "FeatureTrackerBase.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <memory>
#include <opencv2/opencv.hpp>

#include "FeatureDescriptor.h"
#include "FeatureDetectorBase.h"
#include "ORBFeatureDetector.h"
#include "ORBFeatureMatcher.h"

class ORBFeatureTracker : public FeatureTrackerBase {
 public:
  typedef cv::NormTypes DistType;

  ORBFeatureTracker();

  ~ORBFeatureTracker() = default;

  virtual void Init(int width, int height, int max_number_of_features);

  void Init(int width, int height, int max_number_of_features,
            float scale_factor, int number_of_levels, int edge_dist,
            int patch_size, bool refine_corners, DistType dist_type,
            float refine_dist);

  virtual void Track(const cv::Mat& image,
                     Eigen::Matrix2Xf* current_measurements,
                     Eigen::Matrix2Xf* previous_measurements,
                     std::vector<int>* feature_ids);

  virtual void Display();

  inline virtual int GetWidth() const { return width_; }

  inline virtual int GetHeight() const { return height_; }

  inline virtual bool IsInitialized() const { return is_initialized_; }

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

  void GetFeatureDescriptor(FeatureDescriptoru* descriptors);

  void GetFeatureDescriptor(cv::Mat* descriptors) const {
    current_descriptors_.copyTo(*descriptors);
  }

  virtual void GetFeatureLocations(std::vector<Eigen::Vector2d>* points) const {
    CHECK_NOTNULL(points);
    points->resize(current_feature_locations_.size());
    for (size_t i = 0; i < current_feature_locations_.size(); i++) {
      (*points)[i] << current_feature_locations_[i].pt.x,
          current_feature_locations_[i].pt.y;
    }
  }

  virtual const std::vector<int>& GetNewTrackIds() const { return previous_track_ids_; }

  virtual int GetNumberOfFeaturesDetected() const {
    return current_feature_locations_.size();
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

  std::vector<cv::KeyPoint> previous_feature_locations_;
  std::vector<cv::KeyPoint> current_feature_locations_;
  std::vector<cv::DMatch> matches_;
  std::vector<int> previous_track_ids_;
  std::vector<int> current_track_ids_;

  cv::Mat previous_descriptors_;
  cv::Mat current_descriptors_;

  // For debugging we keep the images here
  cv::Mat previous_image_;
  cv::Mat current_image_;
};

#endif
