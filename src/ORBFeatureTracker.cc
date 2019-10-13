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

  previous_feature_locations_.reserve(1000);
  current_feature_locations_.reserve(1000);
  matches_.reserve(1000);
  previous_track_ids_.reserve(1000);
  current_track_ids_.reserve(1000);

  detector_.reset(new ORBFeatureDetector);
  matcher_.reset(new ORBFeatureMatcher);
}

void ORBFeatureTracker::Track(const cv::Mat& image,
                              Eigen::Matrix2Xf* current_measurements,
                              Eigen::Matrix2Xf* previous_measurements,
                              std::vector<int>* feature_ids) {
  std::swap(previous_feature_locations_, current_feature_locations_);
  current_feature_locations_.resize(0);

  std::swap(previous_track_ids_, current_track_ids_);
  cv::swap(current_descriptors_, previous_descriptors_);

  int number_of_features =
      detector_->Detect(image, &current_feature_locations_, &current_descriptors_);
  current_image_ = detector_->GetCvImage();

  previous_image_ = detector_->GetCvImage();

  if (previous_feature_locations_.size() > 0 and number_of_features > 0) {
    matcher_->Match(current_descriptors_, previous_descriptors_, &matches_);
  }
  current_track_ids_.clear();
  current_track_ids_.resize(current_feature_locations_.size(), -1);

  std::vector<cv::DMatch> checked_matches;
  cv::Point2f diff(0., 0.);
  checked_matches.reserve(matches_.size());
  VLOG(4) << "original matches size: " << matches_.size();
  for (const auto& match : matches_) {
    diff = previous_feature_locations_[match.trainIdx].pt -
           current_feature_locations_[match.queryIdx].pt;
    if (std::fabs(diff.x) < width_ and std::fabs(diff.y) < height_) {
      checked_matches.push_back(match);
    }
  }
  VLOG(4) << "checked matches size: " << checked_matches.size();
  std::swap(matches_, checked_matches);

  for (const auto& match : matches_) {
    current_track_ids_[match.queryIdx] = previous_track_ids_[match.trainIdx];
  }

  for (auto& id : current_track_ids_) {
    if (id == -1) {
      id = ++current_track_id_;
    }
  }

  current_measurements->resize(2, matches_.size());
  previous_measurements->resize(2, matches_.size());

  feature_ids->clear();
  for (size_t i = 0; i < matches_.size(); i++) {
    current_measurements->operator()(0, i) =
        current_feature_locations_[matches_[i].queryIdx].pt.x;
    current_measurements->operator()(1, i) =
        current_feature_locations_[matches_[i].queryIdx].pt.y;

    previous_measurements->operator()(0, i) =
        previous_feature_locations_[matches_[i].trainIdx].pt.x;
    previous_measurements->operator()(1, i) =
        previous_feature_locations_[matches_[i].trainIdx].pt.y;

    feature_ids->push_back(current_track_ids_[matches_[i].queryIdx]);
  }
}

void ORBFeatureTracker::GetFeatureDescriptor(FeatureDescriptoru* descriptors) {
  int descriptor_size = current_descriptors_.cols;
  int number_of_features = current_descriptors_.rows;
  descriptors->Configure(descriptor_size, number_of_features);
  descriptors->Resize(number_of_features);
  if (current_descriptors_.isContinuous()) {
    memcpy(descriptors->descriptor(0), current_descriptors_.data,
           descriptor_size * number_of_features);
  } else {
    for (int i = 0; i < number_of_features; i++) {
      memcpy(descriptors->descriptor(i),
             &current_descriptors_.at<uint8_t>(i, 0), descriptor_size);
    }
  }
}

void ORBFeatureTracker::Display() {
  static int iter = 0;
  cv::Mat image_matches;
  if (iter > 0) {
    cv::drawMatches(current_image_, current_feature_locations_, previous_image_,
                    previous_feature_locations_, matches_, image_matches,
                    cv::Scalar(255, 0, 0), cv::Scalar::all(-1),
                    std::vector<char>(), 0);
    cv::imshow("debug_image", image_matches);
    cv::waitKey(1);
    /*  save images
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << iter;
    std::string img_name = "image_" + ss.str() + ".jpg";
    cv::imwrite(img_name, image_matches);
    */
  }
  iter++;
  current_image_.copyTo(previous_image_);
}
