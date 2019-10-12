/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: ORBFeatureMatcher.cc
 *
 *          Created On: Thu 26 Sep 2019 03:27:02 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#include "ORBFeatureMatcher.h"

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <unordered_map>
#include <unordered_set>

DEFINE_int32(dist_type, 4, "1=NORM_L1, 2=NORM_L2, 4=NORM_HAMMING.");
DEFINE_double(refine_dist, 64.0, "Reject distance > refine_dist matches.");

ORBFeatureMatcher::ORBFeatureMatcher()
    : dist_type_(DistType::NORM_HAMMING),
      refine_dist_(0),
      is_initialized_(false) {}

void ORBFeatureMatcher::Init() {
  Init(static_cast<DistType>(FLAGS_dist_type), FLAGS_refine_dist);
}

void ORBFeatureMatcher::Init(DistType dist_type, float refine_dist) {
  CHECK_GT(refine_dist, 0.);
  dist_type_ = dist_type;
  refine_dist_ = refine_dist;
  matches_.reserve(2000);
  refined_matches_.reserve(2000);

  matcher_ = cv::BFMatcher::create(dist_type_);

  is_initialized_ = true;
}

ORBFeatureMatcher::ORBFeatureMatcher(const ORBFeatureMatcher& d) {
  if (d.IsInitialized()) {
    Init(d.GetDistType(), d.GetRefineDist());
  }
}

ORBFeatureMatcher& ORBFeatureMatcher::operator=(const ORBFeatureMatcher& d) {
  if (this == &d) {
    return *this;
  }
  if (d.IsInitialized()) {
    Init(d.GetDistType(), d.GetRefineDist());
  }
  return *this;
}

int ORBFeatureMatcher::Match(const cv::Mat& descriptors_img1,
                             const cv::Mat& descriptors_img2,
                             std::vector<cv::DMatch>* output_matches) {
  if (descriptors_img1.empty() or descriptors_img2.empty()) return -1;

  matches_.resize(0);
  matcher_->match(descriptors_img1, descriptors_img2, matches_);

  std::unordered_set<int> query_indicies;
  std::unordered_set<int> train_indicies;

  VLOG(4) << "Number of features in image 1: " << descriptors_img1.rows
          << ", in image 2: " << descriptors_img2.rows;

  std::unordered_map<int, int> train_index_to_index_map;
  refined_matches_.resize(0);
  for (size_t i = 0; i < matches_.size(); i++) {
    if (matches_[i].distance < refine_dist_) {
      const int& key = matches_[i].trainIdx;
      if (train_index_to_index_map.count(key) == 0) {
        train_index_to_index_map[key] = refined_matches_.size();
        refined_matches_.push_back(matches_[i]);
      } else if (refined_matches_[train_index_to_index_map[key]].distance <
                 matches_[i].distance) {
        refined_matches_[train_index_to_index_map[key]] = matches_[i];
      }
    }
  }
  VLOG(4) << "Number of refined matches: " << refined_matches_.size();
  if (refined_matches_.size() == 0)
    LOG(WARNING) << "Couldn't find refined matches";

  *output_matches = refined_matches_;
  return refined_matches_.size();
}
