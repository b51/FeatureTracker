/*************************************************************************
*
*              Author: b51
*                Mail: b51live@gmail.com
*            FileName: main_track.cc
*
*          Created On: Sun 13 Oct 2019 11:00:50 AM CST
*     Licensed under The MIT License [see LICENSE for details]
*
************************************************************************/

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "FeatureTrackerBase.h"
#include "ORBFeatureTracker.h"
// #include "SuperPointFeatureTracker.h"

DEFINE_string(
    image_dir,
    "/home/b51live/Sources/SuperPointPretrainedNetwork/assets",
    "Where to find images");
DEFINE_string(image_suffix, ".jpg", "Suffix of images, default: .jpg");
DEFINE_int32(max_number_of_features, 500, "max number of detected features");

DEFINE_string(type, "orb", "Feature detector type[orb, sp], default: orb");

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
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;

  std::vector<std::string> image_filenames;
  std::string string_file = FLAGS_image_dir;  // + "/rgb.txt";
  LoadImages(string_file, image_filenames);

  std::cout << "Usage: " << argv[0]
            << " --image_dir path/to/images --suffix .jpg/.png --model"
               " path/to/model --type orb/sp [--model path/to/sp_model --W"
               " sp_input_W --H sp_input_H --CUDA device_num]"
            << std::endl;

  cv::Mat image = cv::imread(image_filenames[0], CV_LOAD_IMAGE_UNCHANGED);

  LOG(INFO) << "type: " << FLAGS_type;
  std::shared_ptr<FeatureTrackerBase> tracker;
  if (FLAGS_type.compare("orb") == 0) {
    tracker = std::make_shared<ORBFeatureTracker>();
  } else if (FLAGS_type.compare("sp") == 0 or
             FLAGS_type.compare("superpoint") == 0) {
    // tracker = std::make_shared<SuperPointFeatureTracker>();
  } else {
    LOG(FATAL) << "Feature type is not supported, only support orb/sp";
  }

  tracker->Init(image.cols, image.rows, FLAGS_max_number_of_features);

  // coordinates in pixel plane
  Eigen::Matrix2Xf current_measurements;
  Eigen::Matrix2Xf previous_measurements;
  std::vector<int> feature_ids;

  for (auto image_filename : image_filenames) {
    image = cv::imread(image_filename, CV_LOAD_IMAGE_UNCHANGED);
    if (image.channels() > 1) {
      cv::cvtColor(image, image, CV_BGR2GRAY);
    }
    if (image.empty()) {
      LOG(FATAL) << "Failed to load image at: " << FLAGS_image_dir << "/"
                 << image_filename;
    }
    tracker->Track(image, &current_measurements, &previous_measurements,
                   &feature_ids);
    tracker->Display();
    VLOG(4) << "Feature size: " << current_measurements.size();
    if (FLAGS_type.compare("orb") == 0) {
      FeatureDescriptoru descriptors;
      std::dynamic_pointer_cast<ORBFeatureTracker>(tracker)
          ->GetFeatureDescriptor(&descriptors);
    } else if (FLAGS_type.compare("sp") == 0 or
               FLAGS_type.compare("superpoint") == 0) {
      // FeatureDescriptorf descriptors;
      // std::dynamic_pointer_cast<SuperPointFeatureTracker>(tracker)
      //     ->GetFeatureDescriptor(&descriptors);
    }
  }
}
