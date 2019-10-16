/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: SuperPointFeatureDetector.h
 *
 *          Created On: Thu 26 Sep 2019 04:07:29 AM UTC
 *     Licensed under The MIT License [see LICENSE for details]
 *
************************************************************************/

#ifndef FEATURE_TRACKER_SUPER_POINT_FEATURE_DETECTOR_H_
#define FEATURE_TRACKER_SUPER_POINT_FEATURE_DETECTOR_H_

#include "FeatureDetectorBase.h"

#include <torch/script.h>
#include <iostream>
#include <memory>
#include <Eigen/Core>
#include <gflags/gflags.h>

class SuperPointFeatureDetector : public FeatureDetectorBase {
 public:
  SuperPointFeatureDetector();
  ~SuperPointFeatureDetector() = default;

  virtual void Init(int width, int height, int max_number_of_features);

  void Init(int width, int height, int max_number_of_features,
            std::string model_path, float conf_thresh, int nms_dist);

  virtual void Detect(const cv::Mat& image,
                      Eigen::Matrix2Xf* current_measurements,
                      FeatureDescriptorf* current_feature_descriptors);

  inline virtual bool IsInitialized() const { return is_initialized_; }

  inline virtual int GetWidth() const { return width_; }

  inline virtual int GetHeight() const { return height_; }

  inline virtual int GetMaxNumberOfFeatures() const {
    return max_number_of_features_;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  /**
   *  Detect input
   *      image: gray image normed to [0.0, 1.0]
   *  return:
   *      feature_locations: 3 X N feature points, [x, y, confidence]^T X N
   *      descriptors: normed 256 X N descriptors
   */
  void Detect(const cv::Mat& image, Eigen::Matrix3Xf& filtered_pts,
              torch::Tensor& descriptors);

  /**
   *  Run a faster approximate Non-Max-Suppression on Eigen corners shaped:
   *    3xN [x_i,y_i,conf_i]^T
   *
   *  Algo summary: Create a grid sized height x width. Assign each corner
   *  location a 1, rest are zeros. Iterate through all the 1's and convert
   *  them either to -1 or 0. Suppress points by setting nearby values to 0.
   *
   *  Grid Value Legend:
   *  -1 : Kept.
   *   0 : Empty or suppressed.
   *   1 : To be processed (converted to either kept or supressed).
   *
   *  NOTE: The NMS first rounds points to integers, so NMS distance might not
   *  be exactly dist_thresh. It also assumes points are within image boundaries.
   *
   *  Inputs
   *    in_corners - 3 x N Eigen Matrix with corners [x_i, y_i, confidence_i]^T.
   *    width - Image width.
   *    height - Image height.
   *    dist_thresh - Distance to suppress, measured as an infinty norm distance.
   *  Returns
   *    filtered_pts - 3 x N Eigen Matrix with surviving corners.
   */
  void NMSFast(const Eigen::Matrix3Xf& pts, int width, int height,
               int dist_thresh, Eigen::Matrix3Xf& filtered_pts);

  int width_;
  int height_;
  int max_number_of_features_;

  std::string model_path_;
  float conf_thresh_;
  int nms_dist_;
  bool is_initialized_;

  std::shared_ptr<torch::jit::script::Module> module_;
  cv::Mat image_;
};

#endif
