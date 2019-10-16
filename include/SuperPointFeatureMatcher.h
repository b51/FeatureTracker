/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: SuperPointFeatureMatcher.h
 *
 *          Created On: Thu 10 Oct 2019 07:48:30 AM UTC
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#ifndef FEATURE_TRACKER_SUPER_POINT_FEATURE_MATCHER_H_
#define FEATURE_TRACKER_SUPER_POINT_FEATURE_MATCHER_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "FeatureDescriptor.h"

class SuperPointFeatureMatcher {
 public:
  /* Feature matched indices. */
  struct FeatureMatch {
    int index1;
    int index2;
    float distance;
    FeatureMatch(int idx1, int idx2, float dist)
        : index1(idx1), index2(idx2), distance(dist){};
    FeatureMatch()
        : index1(-1), index2(-1), distance(std::numeric_limits<float>::max()){};
  };

  SuperPointFeatureMatcher();

  ~SuperPointFeatureMatcher() = default;

  void Init();

  void Init(float nn_thresh);

  bool IsInitialized() const { return is_initialized_; }

  SuperPointFeatureMatcher(const SuperPointFeatureMatcher& d);

  SuperPointFeatureMatcher& operator=(const SuperPointFeatureMatcher& d);

  int Match(const FeatureDescriptorf& descriptors_img1,
            const FeatureDescriptorf& descriptors_img2,
            Eigen::Matrix3Xf* output_matches);

  inline float GetRefineDist() const { return refine_dist_; }

  std::vector<FeatureMatch>& GetMatches() { return matches_; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  int Match(const FeatureDescriptorf& descriptors_img1,
            const FeatureDescriptorf& descriptors_img2);
  float refine_dist_;
  bool is_initialized_;
  std::vector<FeatureMatch> matches_;
};

#endif
