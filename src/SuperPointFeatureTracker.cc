/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: SuperPointFeatureTracker.cc
 *
 *          Created On: Fri 11 Oct 2019 07:52:29 AM UTC
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#include "SuperPointFeatureTracker.h"

SuperPointFeatureTracker::SuperPointFeatureTracker()
    : detector_(nullptr),
      matcher_(nullptr),
      width_(0),
      height_(0),
      current_track_id_(-1),
      is_initialized_(false) {}

void SuperPointFeatureTracker::Init(int width, int height,
                                    int max_number_of_features) {
  LocalInit(width, height);
  detector_->Init(width, height, max_number_of_features);
  matcher_->Init();
  is_initialized_ = true;
}

void SuperPointFeatureTracker::Init(int width, int height,
                                    int max_number_of_features,
                                    std::string model_path, float conf_thresh,
                                    int nms_dist, float nn_thresh) {
  LocalInit(width, height);
  detector_->Init(width, height, max_number_of_features, model_path,
                  conf_thresh, nms_dist);
  matcher_->Init(nn_thresh);
  is_initialized_ = true;
}

void SuperPointFeatureTracker::LocalInit(int width, int height) {
  width_ = width;
  height_ = height;

  previous_track_ids_.reserve(1000);
  current_track_ids_.reserve(1000);

  detector_ = std::make_shared<SuperPointFeatureDetector>();
  matcher_ = std::make_shared<SuperPointFeatureMatcher>();
}

void SuperPointFeatureTracker::GetFeatureLocations(
    std::vector<Eigen::Vector2d>* points) const {
  CHECK_NOTNULL(points);
  points->resize(current_feature_locations_.cols());
  for (int i = 0; i < current_feature_locations_.cols(); i++) {
    (*points)[i] << current_feature_locations_.col(i)[0],
        current_feature_locations_.col(i)[1];
  }
}

void SuperPointFeatureTracker::Track(const cv::Mat& image,
                                     Eigen::Matrix2Xf* current_measurements,
                                     Eigen::Matrix2Xf* previous_measurements,
                                     std::vector<int>* feature_ids) {
  previous_feature_locations_ = current_feature_locations_;
  image.copyTo(current_image_);

  std::swap(previous_track_ids_, current_track_ids_);
  current_descriptors_.Swap(&previous_descriptors_);

  detector_->Detect(image, &current_feature_locations_, &current_descriptors_);
  int number_of_features = current_feature_locations_.cols();
  if (previous_feature_locations_.cols() > 0 and number_of_features > 0) {
    matcher_->Match(current_descriptors_, previous_descriptors_, &matches_);
  }
  int number_of_matches = matches_.cols();
  current_track_ids_.clear();
  current_track_ids_.resize(number_of_features, -1);
  for (int i = 0; i < number_of_matches; i++) {
    current_track_ids_[matches_.col(i)[0]] = previous_track_ids_[matches_.col(i)[1]];
  }
  for (auto& id : current_track_ids_) {
    if (id == -1) {
      id = ++current_track_id_;
    }
  }

  current_measurements->resize(2, number_of_matches);
  previous_measurements->resize(2, number_of_matches);
  feature_ids->clear();

  for (int i = 0; i < number_of_matches; i++) {
    current_measurements->col(i) = current_feature_locations_.col(matches_.col(i)[0]);
    previous_measurements->col(i) = previous_feature_locations_.col(matches_.col(i)[1]);
    feature_ids->push_back(current_track_ids_[matches_.col(i)[0]]);
  }
}

void SuperPointFeatureTracker::Display() {
  cv::Mat image_matches;
  std::vector<cv::KeyPoint> previous_cv_keypoints;
  std::vector<cv::KeyPoint> current_cv_keypoints;
  std::vector<cv::DMatch> cv_matches;
  int number_of_matches = matches_.cols();
  if (number_of_matches > 0) {
    for (int i = 0; i < number_of_matches; i++) {
      int x = std::round(current_feature_locations_.col(matches_.col(i)[0])[0]);
      int y = std::round(current_feature_locations_.col(matches_.col(i)[0])[1]);
      int px = std::round(previous_feature_locations_.col(matches_.col(i)[0])[0]);
      int py = std::round(previous_feature_locations_.col(matches_.col(i)[1])[1]);
      // int id = current_track_ids_[matches_.col(i)[0]];

      previous_cv_keypoints.emplace_back(cv::KeyPoint(px, py, 1.));
      current_cv_keypoints.emplace_back(cv::KeyPoint(x, y, 1.));
      cv_matches.emplace_back(i, i, matches_.col(i)[2]);
      //          cv::Scalar(0, 255, 255));
      // cv::putText(current_image_, std::to_string(id), cv::Point(px, py), 1, 1.0,
      //             cv::Scalar(0, 0, 0), 1, 1, false);
    }
    cv::drawMatches(current_image_, current_cv_keypoints, previous_image_,
                    previous_cv_keypoints, cv_matches, image_matches,
                    cv::Scalar(255, 0, 0), cv::Scalar::all(-1),
                    std::vector<char>(), 0);
    cv::imshow("tracker", image_matches);
    cv::waitKey(1);
    /* save images
    static int iter = 0;
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << iter++;
    std::string img_name = "image_" + ss.str() + ".jpg";
    cv::imwrite(img_name, image_matches);
    */
  }
  current_image_.copyTo(previous_image_);
}
