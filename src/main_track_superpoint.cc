/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: main_track_superpoint.cc
 *
 *          Created On: Sun 29 Sep 2019 07:33:42 AM UTC
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "FeatureDescriptor.h"
#include "FeatureDetectorBase.h"
#include "ORBFeatureDetector.h"
#include "SuperPointFeatureDetector.h"
#include "SuperPointFeatureMatcher.h"
#include "SuperPointFeatureTracker.h"

DEFINE_string(
    image_dir,
    "/home/b51live/Sources/SuperPointPretrainedNetwork/assets",
    "Where to find images");
DEFINE_string(image_suffix, ".jpg", "Suffix of images, default: .jpg");
DEFINE_int32(max_number_of_features, 500, "max number of detected features");
DECLARE_string(model_path);

/*
void LoadImages(const std::string& file_name,
                std::vector<std::string>& image_filenames) {
  std::ifstream f;
  f.open(file_name.c_str());
  if (!f.good()) LOG(FATAL) << file_name << " not exists";

  while (!f.eof()) {
    std::string s;
    getline(f, s);
    if (!s.empty()) {
      std::stringstream ss;
      ss << s;
      std::string image_filename;
      ss >> image_filename;
      image_filenames.push_back(image_filename);
    }
  }
}
*/

void LoadImages(const std::string& dir_name,
                std::vector<std::string>& image_filenames) {
  std::vector<cv::String> files;
  cv::glob(dir_name, files);
  for (auto file : files) {
    std::string image_file(file);
    size_t index = image_file.rfind(".");
    if (index != std::string::npos) {
      std::string suffix = image_file.substr(index);
      if (suffix.compare(FLAGS_image_suffix) == 0) {
        image_filenames.emplace_back(image_file);
      }
    }
  }  // end of for loop
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_minloglevel = google::INFO;
  FLAGS_v = 4;
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;

  std::cout << "Usage: " << argv[0]
            << " --image_dir path/to/images --model_path path/to/model"
            << std::endl;

  std::vector<std::string> image_filenames;
  std::string string_file = FLAGS_image_dir;  // + "/rgb.txt";
  LoadImages(string_file, image_filenames);

  VLOG(3) << "Start processing sequence ...";
  VLOG(3) << "Images in the sequence: " << image_filenames.size();

  cv::Mat image = cv::imread(image_filenames[0], CV_LOAD_IMAGE_UNCHANGED);

  /* Detect image stream
  std::shared_ptr<FeatureDetectorBase> detector;
  detector = std::make_shared<SuperPointFeatureDetector>();
  detector->Init(image.cols, image.rows, FLAGS_max_number_of_features);

  // coordinates in pixel plane
  Eigen::Matrix2Xd measurements;
  std::vector<float> feature_orientations;
  std::vector<float> feature_scales;
  FeatureDescriptorf feature_descriptors;

  for (auto image_filename : image_filenames) {
    image = cv::imread(FLAGS_image_dir + "/" + image_filename,
                       CV_LOAD_IMAGE_UNCHANGED);
    if (image.channels() > 1) cv::cvtColor(image, image, CV_BGR2GRAY);
    if (image.empty()) {
      LOG(FATAL) << "Failed to load image at: " << FLAGS_image_dir << "/"
                 << image_filename;
    }
    VLOG(4) << "***** " << image_filename << " *****" << std::endl;
    detector->Detect(image, &measurements, &feature_orientations,
                     &feature_scales, &feature_descriptors);
    VLOG(4) << "Feature size: " << feature_scales.size();
  }
  */

  /*  Detect two images and match
  std::shared_ptr<SuperPointFeatureMatcher> matcher;
  matcher = std::make_shared<SuperPointFeatureMatcher>();
  matcher->Init();

  cv::Mat image0 = cv::imread(image_filenames[0], CV_LOAD_IMAGE_UNCHANGED);
  Eigen::Matrix2Xd measurements0;
  FeatureDescriptorf feature_descriptors0;
  detector->Detect(image0, &measurements0, &feature_descriptors0);

  cv::Mat image1 = cv::imread(image_filenames[1], CV_LOAD_IMAGE_UNCHANGED);
  Eigen::Matrix2Xd measurements1;
  FeatureDescriptorf feature_descriptors1;
  detector->Detect(image1, &measurements1, &feature_descriptors1);
  Eigen::Matrix3Xd matches;
  matcher->Match(feature_descriptors0, feature_descriptors1, &matches);
  */

  /*  Track two Images */
  std::shared_ptr<SuperPointFeatureTracker> tracker;
  tracker = std::make_shared<SuperPointFeatureTracker>();
  tracker->Init(image.cols, image.rows, FLAGS_max_number_of_features);

  Eigen::Matrix2Xf current_measurements;
  Eigen::Matrix2Xf previous_measurements;
  std::vector<int> feature_ids;
  for (auto image_filename : image_filenames) {
    image = cv::imread(image_filename, CV_LOAD_IMAGE_UNCHANGED);
    if (image.channels() > 1) cv::cvtColor(image, image, CV_BGR2GRAY);
    if (image.empty()) {
      LOG(FATAL) << "Failed to load image at: " << image_filename;
    }
    VLOG(4) << "***** " << image_filename << " *****" << std::endl;
    tracker->Track(image, &current_measurements, &previous_measurements,
                   &feature_ids);
    tracker->Display();
    /*
    for (size_t ii = 0; ii < feature_ids.size(); ii++)
      LOG(INFO) << "id: " << feature_ids[ii] << ", ["
                << previous_measurements.col(ii)[0] << ", "
                << previous_measurements.col(ii)[1] << "], ["
                << current_measurements.col(ii)[0] << ", "
                << current_measurements.col(ii)[1] << " ]";
    */
    // detector->Detect(image, &measurements, &feature_orientations,
    //                 &feature_scales, &feature_descriptors);
    VLOG(5) << "Feature size: " << current_measurements.size();
  }

  return 0;
}
