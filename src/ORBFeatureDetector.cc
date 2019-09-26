/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: ORBFeatureDetector.cc
 *
 *          Created On: Thu 27 Jun 2019 03:26:58 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#include "ORBFeatureDetector.h"

#include <glog/logging.h>

DEFINE_double(scale_factor, 1.1f, "Scale increment");
DEFINE_int32(number_of_levels, 8, "Number of scales.");
DEFINE_int32(edge_dist, 5, "Distance to edge not considered.");
DEFINE_int32(patch_size, 31, "Patch size.");
DEFINE_bool(refine_corners, false, "Whether we want to refine corners");

const static unsigned int kDescriptorLengthInBytes = 32;

ORBFeatureDetector::ORBFeatureDetector()
    : width_(0),
      height_(0),
      max_number_of_features_(0),
      scale_factor_(0.),
      number_of_levels_(0),
      edge_dist_(0),
      patch_size_(0),
      refine_corners_(false),
      is_initialized_(false) {}

void ORBFeatureDetector::Init(int width, int height,
                              int max_number_of_features) {
  CHECK_GT(width, 0);
  CHECK_GT(height, 0);
  CHECK_GT(max_number_of_features, 0);
  Init(width, height, max_number_of_features, FLAGS_scale_factor,
       FLAGS_number_of_levels, FLAGS_edge_dist, FLAGS_patch_size,
       FLAGS_refine_corners);
}

void ORBFeatureDetector::Init(int width, int height, int max_number_of_features,
                              float scale_factor, int number_of_levels,
                              int edge_dist, int patch_size,
                              bool refine_corners) {
  CHECK_GT(width, 0);
  CHECK_GT(height, 0);
  CHECK_GT(max_number_of_features, 0);
  CHECK_GT(scale_factor, 0);
  CHECK_GT(number_of_levels, 0);
  CHECK_GT(edge_dist, 0);
  CHECK_GT(patch_size, 0);

  width_ = width;
  height_ = height;
  max_number_of_features_ = max_number_of_features;
  scale_factor_ = scale_factor;
  number_of_levels_ = number_of_levels;
  edge_dist_ = edge_dist;
  patch_size_ = patch_size;
  refine_corners_ = refine_corners;

  image_.create(height_, width_, CV_8UC1);
  mask_ = cv::Mat(cv::Size(width_, height_), CV_8UC1, 0xFF);
  VLOG(3) << "ORB detector parameters :";
  VLOG(3) << "      max_number_of_features: " << max_number_of_features_;
  VLOG(3) << "      scale_factor: " << scale_factor_;
  VLOG(3) << "      number_of_levels: " << number_of_levels_;
  VLOG(3) << "      edge_dist: " << edge_dist_;
  detector_ =
      cv::ORB::create(max_number_of_features_, scale_factor_, number_of_levels_,
                      edge_dist_, 0, 2, cv::ORB::HARRIS_SCORE, patch_size_);
  is_initialized_ = true;
}

ORBFeatureDetector::ORBFeatureDetector(const ORBFeatureDetector& d)
    : FeatureDetectorBase() {
  if (d.IsInitialized()) {
    Init(d.GetWidth(), d.GetHeight(), d.GetMaxNumberOfFeatures(),
         d.GetScaleFactor(), d.GetNumberOfLevels(), d.GetEdgeDist(),
         d.GetPatchSize(), d.GetRefineCorners());
  }
}

ORBFeatureDetector& ORBFeatureDetector::operator=(const ORBFeatureDetector& d) {
  if (this == &d) {
    return *this;
  }
  if (d.IsInitialized()) {
    Init(d.GetWidth(), d.GetHeight(), d.GetMaxNumberOfFeatures(),
         d.GetScaleFactor(), d.GetNumberOfLevels(), d.GetEdgeDist(),
         d.GetPatchSize(), d.GetRefineCorners());
  }
  return *this;
}

void ORBFeatureDetector::SetMask(const cv::Mat& mat) {
  CHECK_EQ(mat.cols, mask_.cols);
  CHECK_EQ(mat.rows, mask_.rows);
  mask_ = mat;
}

void ORBFeatureDetector::SetNumberOfLevels(int number_of_levels) {
  if (number_of_levels != number_of_levels_) {
    CHECK(IsInitialized()) << "Please initialize first";
    number_of_levels_ = number_of_levels;
    detector_ = cv::ORB::create(max_number_of_features_, scale_factor_,
                                number_of_levels_, edge_dist_, 0, 2,
                                cv::ORB::HARRIS_SCORE, patch_size_);
  }
}

void ORBFeatureDetector::Refine(const cv::Mat& image,
                                std::vector<cv::KeyPoint>* feature_locations) {
  CHECK_NOTNULL(feature_locations);
  CHECK(IsInitialized()) << "Please initialize first";
  std::vector<cv::Point2f> locs;
  locs.resize(feature_locations->size());
  for (size_t i = 0; i < feature_locations->size(); i++) {
    locs[i] = (*feature_locations)[i].pt;
  }
  cv::TermCriteria criteria =
      cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
  cv::cornerSubPix(image, locs, cv::Size(5, 5), cv::Size(-1, -1), criteria);
  for (size_t i = 0; i < feature_locations->size(); i++) {
    (*feature_locations)[i].pt = locs[i];
  }
}

int ORBFeatureDetector::Detect(const cv::Mat& image,
                               std::vector<cv::KeyPoint>* feature_locations,
                               cv::Mat* descriptors) {
  CHECK(IsInitialized());
  CHECK_NOTNULL(descriptors);
  CHECK_NOTNULL(feature_locations);
  CHECK(!image.empty()) << "Empty image.";
  CHECK_EQ(image.rows, height_);
  CHECK_EQ(image.cols, width_) << "Size change on the fly is not considered.";
  image.copyTo(image_);

  int number_of_features = -1;

  VLOG(3) << "Detect original size image";
  feature_locations->resize(0);
  VLOG(3) << image.cols << " " << image.rows;
  detector_->detectAndCompute(image, mask_, *feature_locations, *descriptors);
  number_of_features = feature_locations->size();
  VLOG(3) << "number_of_features: " << number_of_features;

  if (refine_corners_) {
    Refine(image, feature_locations);
  }
  return number_of_features;
}

void ORBFeatureDetector::Detect(
    const cv::Mat& image, Eigen::Matrix2Xd* current_measurements,
    std::vector<float>* current_feature_orientations,
    std::vector<float>* current_feature_scales,
    FeatureDescriptor* current_feature_descriptors) {
  CHECK_NOTNULL(current_measurements);
  CHECK_NOTNULL(current_feature_orientations);
  CHECK_NOTNULL(current_feature_scales);
  CHECK_NOTNULL(current_feature_descriptors);
  std::vector<cv::KeyPoint> feature_locations;
  cv::Mat descriptors;
  int number_of_features = Detect(image, &feature_locations, &descriptors);
  CHECK_EQ(descriptors.cols, kDescriptorLengthInBytes);

  // Prepare output
  current_measurements->resize(2, number_of_features);
  for (int i = 0; i < number_of_features; ++i) {
    current_measurements->operator()(0, i) = feature_locations[i].pt.x;
    current_measurements->operator()(1, i) = feature_locations[i].pt.y;
  }

  current_feature_orientations->clear();
  current_feature_orientations->reserve(number_of_features);
  current_feature_scales->clear();

  std::vector<float> all_scales;
  for (int i = 0; i < number_of_levels_; ++i) {
    all_scales.push_back(std::pow(scale_factor_, i));
  }
  current_feature_scales->resize(number_of_features, -1.);
  current_feature_descriptors->Configure(kDescriptorLengthInBytes,
                                         number_of_features);
  current_feature_descriptors->Resize(number_of_features);

  // Assign orientation and scale
  for (int i = 0; i < number_of_features; ++i) {
    (*current_feature_orientations)[i] = feature_locations[i].angle;
    CHECK_LT(feature_locations[i].octave, number_of_levels_);
    (*current_feature_scales)[i] = all_scales[feature_locations[i].octave];
  }

  CHECK_EQ(descriptors.type(), CV_8UC1);

  if (descriptors.isContinuous()) {
    CHECK_EQ(descriptors.at<uint8_t>(0, 1), descriptors.data[1])
        << " not row major";
    // if continuous and row major
    memcpy(current_feature_descriptors->descriptor(0), descriptors.data,
           kDescriptorLengthInBytes * number_of_features);
  } else {
    for (int i = 0; i < number_of_features; ++i) {
      memcpy(current_feature_descriptors->descriptor(i),
             &descriptors.at<uint8_t>(i, 0), kDescriptorLengthInBytes);
    }
  }
  // for (int i = 0; i < number_of_features; i++) {
  //   cv::circle(image,
  //              cv::Point(current_measurements->operator()(0, i),
  //                        current_measurements->operator()(1, i)),
  //              5, cv::Scalar(0, 255, 255));
  // }
  // cv::imshow("features", image);
  // cv::waitKey(10);
}
