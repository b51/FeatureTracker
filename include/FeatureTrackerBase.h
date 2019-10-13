/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: FeatureTrackerBase.h
 *
 *          Created On: Sat 12 Oct 2019 08:23:14 AM UTC
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#ifndef FEATURE_TRACKER_FEATURE_TRACKER_BASE_H_
#define FEATURE_TRACKER_FEATURE_TRACKER_BASE_H_

#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>

class FeatureTrackerBase {
 public:
  FeatureTrackerBase(){};

  virtual ~FeatureTrackerBase(){};

  virtual void Init(int width, int height, int max_number_of_features) = 0;

  virtual void Track(const cv::Mat& image,
                     Eigen::Matrix2Xf* current_measurements,
                     Eigen::Matrix2Xf* previous_measurements,
                     std::vector<int>* feature_ids) = 0;

  virtual bool IsInitialized() const = 0;

  inline virtual int GetWidth() const = 0;

  inline virtual int GetHeight() const = 0;

  virtual void GetFeatureLocations(
      std::vector<Eigen::Vector2d>* points) const = 0;

  virtual const std::vector<int>& GetNewTrackIds() const = 0;

  virtual int GetNumberOfFeaturesMatched() const = 0;

  virtual void Display() = 0;

 private:
  FeatureTrackerBase(const FeatureTrackerBase& ft) = delete;
  FeatureTrackerBase& operator=(const FeatureTrackerBase& ft) = delete;
};
#endif
