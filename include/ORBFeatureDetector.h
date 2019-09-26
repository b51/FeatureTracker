/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: ORBFeatureDetector.h
 *
 *          Created On: Wed 25 Sep 2019 04:26:40 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#ifndef FEATURE_TRACKER_ORB_FEATURE_DETECTOR_H_
#define FEATURE_TRACKER_ORB_FEATURE_DETECTOR_H_

#include "FeatureDetectorBase.h"

#include <opencv2/opencv.hpp>

class ORBFeatureDetector : public FeatureDetectorBase {
 public:
  ORBFeatureDetector();
  ~ORBFeatureDetector() = default;

  virtual void Init(int width, int height, int max_number_of_features);

  void Init(int width, int height, int max_number_of_features,
            float scale_factor, int number_of_levels, int edge_dist,
            int patch_size, bool refine_corners);

  virtual void Detect(const cv::Mat& image,
                      Eigen::Matrix2Xd* current_measurements,
                      std::vector<float>* current_feature_orientations,
                      std::vector<float>* current_feature_scales,
                      FeatureDescriptor* current_feature_descriptors);

  int Detect(const cv::Mat& image, std::vector<cv::KeyPoint>* feature_locations,
             cv::Mat* descriptors);

  inline virtual bool IsInitialized() const { return is_initialized_; }

  inline virtual int GetWidth() const { return width_; }

  inline virtual int GetHeight() const { return height_; }

  inline int GetMaxNumberOfFeatures() const { return max_number_of_features_; }

  inline float GetScaleFactor() const { return scale_factor_; }

  inline int GetNumberOfLevels() const { return number_of_levels_; }

  inline int GetEdgeDist() const { return edge_dist_; }

  inline int GetPatchSize() const { return patch_size_; }

  inline bool GetRefineCorners() const { return refine_corners_; }

  void SetMask(const cv::Mat& mask);

  void SetNumberOfLevels(int number_of_levels);

  ORBFeatureDetector(const ORBFeatureDetector& d);

  ORBFeatureDetector& operator=(const ORBFeatureDetector& d);

  void Refine(const cv::Mat& image,
              std::vector<cv::KeyPoint>* feature_locations);

 private:
  int width_;
  int height_;

  int max_number_of_features_;

  float scale_factor_;
  int number_of_levels_;
  int edge_dist_;
  int patch_size_;
  bool refine_corners_;
  float downsample_scale_;
  bool is_initialized_;

  cv::Mat image_;
  cv::Mat mask_;

  cv::Ptr<cv::ORB> detector_;
};

#endif
