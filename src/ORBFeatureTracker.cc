/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: ORBFeatureTracker.cc
 *
 *          Created On: Thu 26 Sep 2019 04:06:52 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#include "ORBFeatureTracker.h"

ORBFeatureTracker::ORBFeatureTracker()
    : detector_(nullptr),
      matcher_(nullptr),
      width_(0),
      height_(0),
      current_track_id_(-1),
      is_initialized_(false) {}

void ORBFeatureTracker::Init(int width, int height,
                             int max_number_of_features) {
  LocalInit(width, height);
  detector_->Init(width, height, max_number_of_features);
  matcher_->Init();
  is_initialized_ = true;
}

void ORBFeatureTracker::Init(int width, int height, int max_number_of_features,
                             float scale_factor, int number_of_levels,
                             int edge_dist, int patch_size, bool refine_corners,
                             DistType dist_type, float refine_dist) {
  LocalInit(width, height);
  detector_->Init(width, height, max_number_of_features, scale_factor,
                  number_of_levels, edge_dist, patch_size, refine_corners);
  matcher_->Init(dist_type, refine_dist);
  is_initialized_ = true;
}

void ORBFeatureTracker::LocalInit(int width, int height) {
  width_ = width;
  height_ = height;

  prev_feature_locations_.reserve(1000);
  curr_feature_locations_.reserve(1000);
  matches_.reserve(1000);
  prev_track_ids_.reserve(1000);
  curr_track_ids_.reserve(1000);

  detector_.reset(new ORBFeatureDetector);
  matcher_.reset(new ORBFeatureMatcher);
}

void ORBFeatureTracker::Track(const cv::Mat& image,
                              Eigen::Matrix2Xd* curr_measurements,
                              Eigen::Matrix2Xd* prev_measurements,
                              std::vector<int>* feature_ids) {
  std::swap(prev_feature_locations_, curr_feature_locations_);
  curr_feature_locations_.resize(0);

  std::swap(prev_track_ids_, curr_track_ids_);
  cv::swap(curr_descriptor_, prev_descriptor_);

  int number_of_features =
      detector_->Detect(image, &curr_feature_locations_, &curr_descriptor_);
  curr_image_ = detector_->GetCvImage();

  prev_image_ = detector_->GetCvImage();

  if (prev_feature_locations_.size() > 0 and number_of_features > 0) {
    matcher_->Match(curr_descriptor_, prev_descriptor_, &matches_);
  }
  curr_track_ids_.clear();
  curr_track_ids_.resize(curr_feature_locations_.size(), -1);

  std::vector<cv::DMatch> checked_matches;
  cv::Point2f diff(0., 0.);
  checked_matches.reserve(matches_.size());
  VLOG(4) << "original matches size: " << matches_.size();
  for (const auto& match : matches_) {
    diff = prev_feature_locations_[match.trainIdx].pt -
           curr_feature_locations_[match.queryIdx].pt;
    if (std::fabs(diff.x) < width_ and std::fabs(diff.y) < height_) {
      checked_matches.push_back(match);
    }
  }
  VLOG(4) << "checked matches size: " << checked_matches.size();
  std::swap(matches_, checked_matches);

  for (const auto& match : matches_) {
    curr_track_ids_[match.queryIdx] = prev_track_ids_[match.trainIdx];
  }

  for (auto& id : curr_track_ids_) {
    if (id == -1) {
      id = ++current_track_id_;
    }
  }

  curr_measurements->resize(2, matches_.size());
  prev_measurements->resize(2, matches_.size());

  feature_ids->clear();
  for (size_t i = 0; i < matches_.size(); i++) {
    curr_measurements->operator()(0, i) =
        curr_feature_locations_[matches_[i].queryIdx].pt.x;
    curr_measurements->operator()(1, i) =
        curr_feature_locations_[matches_[i].queryIdx].pt.y;

    prev_measurements->operator()(0, i) =
        prev_feature_locations_[matches_[i].trainIdx].pt.x;
    prev_measurements->operator()(1, i) =
        prev_feature_locations_[matches_[i].trainIdx].pt.y;

    feature_ids->push_back(curr_track_ids_[matches_[i].queryIdx]);
  }
  if (prev_descriptor_.rows > 0 && number_of_features > 0) {
    Display();
  }
  curr_image_.copyTo(prev_image_);
}

void ORBFeatureTracker::Display() {
  static int iter = 0;
  cv::Mat image_matches;
  if (iter > 0) {
    cv::Mat image_matches;
    cv::drawMatches(curr_image_, curr_feature_locations_, prev_image_,
                    prev_feature_locations_, matches_, image_matches,
                    cv::Scalar(255, 0, 0), cv::Scalar::all(-1),
                    std::vector<char>(), 0);
    cv::imshow("debug_image", image_matches);
  }
  iter++;
  cv::waitKey(1);
}
