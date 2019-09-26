/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: ORBFeatureMatcher.h
 *
 *          Created On: Thu 26 Sep 2019 03:29:02 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#ifndef FEATURE_TRACKER_ORB_FEATURE_MATCHER_H_
#define FEATURE_TRACKER_ORB_FEATURE_MATCHER_H_

#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

class ORBFeatureMatcher {
 public:
  typedef cv::NormTypes DistType;

  ORBFeatureMatcher();

  ~ORBFeatureMatcher() = default;

  void Init();

  void Init(DistType dist_type, float refine_dist);

  bool IsInitialized() const { return is_initialized_; }

  ORBFeatureMatcher(const ORBFeatureMatcher& d);

  ORBFeatureMatcher& operator=(const ORBFeatureMatcher& d);

  int Match(const cv::Mat& descriptors_img1, const cv::Mat& descriptors_img2,
            std::vector<cv::DMatch>* output_matches);

  inline DistType GetDistType() const { return dist_type_; }

  inline float GetRefineDist() const { return refine_dist_; }

  void SetDistType(DistType dist_type);

  inline void SetRefineDist(float refine_dist) { refine_dist_ = refine_dist; }

  std::vector<cv::DMatch>& GetMatches() { return matches_; }

  std::vector<cv::DMatch>& GetRefinedMatches() { return refined_matches_; }

 private:
  cv::Ptr<cv::BFMatcher> matcher_;
  DistType dist_type_;
  float refine_dist_;
  bool is_initialized_;
  std::vector<cv::DMatch> matches_;
  std::vector<cv::DMatch> refined_matches_;
};

#endif
