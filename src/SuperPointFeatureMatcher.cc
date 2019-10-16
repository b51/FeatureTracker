/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: SuperPointFeatureMatcher.cc
 *
 *          Created On: Thu 10 Oct 2019 07:47:43 AM UTC
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#include "SuperPointFeatureMatcher.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_double(
    nn_thresh, 0.7,
    "Optional descriptor distance below which is a good match (default: 0.7)");

SuperPointFeatureMatcher::SuperPointFeatureMatcher()
  : is_initialized_(false) {}

void SuperPointFeatureMatcher::Init() {
  Init(FLAGS_nn_thresh);
}

void SuperPointFeatureMatcher::Init(float nn_thresh) {
  CHECK_GT(nn_thresh, 0.0);
  refine_dist_ = nn_thresh;
  matches_.reserve(2000);
  is_initialized_ = true;
}

SuperPointFeatureMatcher::SuperPointFeatureMatcher(
    const SuperPointFeatureMatcher& d) {
  if (d.IsInitialized()) {
    Init(d.GetRefineDist());
  }
}

SuperPointFeatureMatcher& SuperPointFeatureMatcher::operator=(
    const SuperPointFeatureMatcher& d) {
  if (this == &d) {
    return *this;
  }
  if (d.IsInitialized()) {
    Init(d.GetRefineDist());
  }
  return *this;
}

int SuperPointFeatureMatcher::Match(const FeatureDescriptorf& descriptors_img1,
                                    const FeatureDescriptorf& descriptors_img2,
                                    Eigen::Matrix3Xf* output_matches) {
  if (descriptors_img1.Empty() or descriptors_img2.Empty()) return -1;

  int number_of_matches = Match(descriptors_img1, descriptors_img2);

  output_matches->resize(3, number_of_matches);
#pragma omp parallel for
  for (int i = 0; i < number_of_matches; i++) {
    output_matches->col(i)[0] = matches_[i].index1;
    output_matches->col(i)[1] = matches_[i].index2;
    output_matches->col(i)[2] = matches_[i].distance;
  }
  VLOG(4) << " number_of_matches " << output_matches->cols();
  return number_of_matches;
}

int SuperPointFeatureMatcher::Match(
    const FeatureDescriptorf& descriptors_img1,
    const FeatureDescriptorf& descriptors_img2) {
  matches_.resize(0);
  // descriptors_img1 size = N1 X 256
  // descriptors_img2 size = N2 X 256
  int number_of_features_1 = descriptors_img1.Size();
  int number_of_features_2 = descriptors_img2.Size();
  int descriptor_size = descriptors_img1.DescriptorSize();

  Eigen::MatrixXf eig_descriptors_1, eig_descriptors_2;
  eig_descriptors_1.resize(number_of_features_1, descriptor_size);
  eig_descriptors_2.resize(number_of_features_2, descriptor_size);

  for (int i = 0; i < number_of_features_1; i++) {
    eig_descriptors_1.row(i) = Eigen::Map<const Eigen::MatrixXf>(
        descriptors_img1.descriptor(i), 1, descriptor_size);
  }
  for (int i = 0; i < number_of_features_2; i++) {
    eig_descriptors_2.row(i) = Eigen::Map<const Eigen::MatrixXf>(
        descriptors_img2.descriptor(i), 1, descriptor_size);
  }

  // nn_matches size = N1 X N2
  Eigen::MatrixXf nn_matches =
      eig_descriptors_1 * eig_descriptors_2.transpose();
  // clip scores to range [-1.0, 1.0]
  nn_matches = 2.0 * nn_matches.array().max(-1.0).min(1.0);
  // make sure scores positive
  nn_matches = 2.0 - nn_matches.array();
  // get sqrt root of scores
  nn_matches = nn_matches.array().sqrt();

  // get min scores of each row
  Eigen::VectorXf scores(nn_matches.rows(), 1);
  scores = nn_matches.rowwise().minCoeff();

  // get each min scores index of rows
  Eigen::VectorXf indices_1(nn_matches.rows(), 1);
  for (int i = 0; i < nn_matches.rows(); i++)
    nn_matches.row(i).minCoeff(&indices_1(i));
  // get each min scores index of cols
  Eigen::VectorXf indices_2(nn_matches.cols(), 1);
  for (int i = 0; i < nn_matches.cols(); i++)
    nn_matches.col(i).minCoeff(&indices_2(i));

  // Threshold the NN matches, meet condition true, else false
  Eigen::Matrix<bool, Eigen::Dynamic, 1> keep(scores.rows(), 1);
  keep = (scores.array() < refine_dist_);

  // Check if nearest neighbor goes both directions and keep those.
#pragma omp parallel for
  for (int i = 0; i < indices_1.rows(); i++) {
    keep[i] = (indices_2[indices_1[i]] == i) && keep[i];
    if (keep[i]) {
      matches_.emplace_back(FeatureMatch(i, indices_1[i], scores[i]));
    }
  }
  return matches_.size();
}
