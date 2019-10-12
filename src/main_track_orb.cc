/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: main_track_orb.cc
 *
 *          Created On: Wed 25 Sep 2019 04:25:40 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "ORBFeatureTracker.h"

DEFINE_string(
    image_dir,
    "/home/ubuntu/Documents/VioBag/downloads/rgbd_dataset_freiburg2_desk",
    "Where to find images");
DEFINE_string(image_suffix, ".jpg", "Suffix of images, default: .jpg");
DEFINE_int32(max_number_of_features, 500, "max number of detected features");

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
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;

  std::vector<std::string> image_filenames;
  std::string string_file = FLAGS_image_dir;// + "/rgb.txt";
  LoadImages(string_file, image_filenames);

  LOG(INFO) << "Start processing sequence ...";
  LOG(INFO) << "Images in the sequence: " << image_filenames.size();

  // cv::Mat image = cv::imread(FLAGS_image_dir + "/" + image_filenames[0],
  //                            CV_LOAD_IMAGE_UNCHANGED);
  cv::Mat image = cv::imread(image_filenames[0], CV_LOAD_IMAGE_UNCHANGED);

  std::shared_ptr<ORBFeatureTracker> tracker;
  tracker = std::make_shared<ORBFeatureTracker>();
  tracker->Init(image.cols, image.rows, FLAGS_max_number_of_features);

  // coordinates in pixel plane
  Eigen::Matrix2Xd current_measurements;
  Eigen::Matrix2Xd previous_measurements;
  std::vector<int> feature_ids;
  int i = 0;

  for (auto image_filename : image_filenames) {
    image = cv::imread(image_filename, CV_LOAD_IMAGE_UNCHANGED);
    if (image.channels() > 1) cv::cvtColor(image, image, CV_BGR2GRAY);
    if (image.empty()) {
      LOG(FATAL) << "Failed to load image at: " << FLAGS_image_dir << "/"
                 << image_filename;
    }
    tracker->Track(image, &current_measurements, &previous_measurements,
                   &feature_ids);
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
    i++;
  }
}
