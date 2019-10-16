/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: SuperPointFeatureDetector.cc
 *
 *          Created On: Fri 27 Sep 2019 08:30:41 AM UTC
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#include "SuperPointFeatureDetector.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Core>
#include <cmath>

DEFINE_string(model, "/home/ubuntu/Models",
              "Path to pytorch model");
DEFINE_double(conf_thresh, 0.015,
              "Detector confidence threshold (default: 0.015)");
DEFINE_int32(H, 120, "Net input image height");
DEFINE_int32(W, 160, "Net input image width");
DEFINE_int32(nms_dist, 4, "Non Maximum Suppression (NMS) distance (default: 4).");
DEFINE_int32(CUDA, 0, "Default GPU device");

//Size of each output cell, Keep this fixed
const int CELL = 8;

SuperPointFeatureDetector::SuperPointFeatureDetector()
    : width_(0),
      height_(0),
      max_number_of_features_(0),
      conf_thresh_(0.),
      nms_dist_(0),
      is_initialized_(false),
      module_(nullptr) {}

void SuperPointFeatureDetector::Init(int width, int height,
                                     int max_number_of_features) {
  CHECK_GT(width, 0);
  CHECK_GT(height, 0);
  CHECK_GT(max_number_of_features, 0);
  Init(width, height, max_number_of_features, FLAGS_model,
       FLAGS_conf_thresh, FLAGS_nms_dist);
}

void SuperPointFeatureDetector::Init(int width, int height,
                                     int max_number_of_features,
                                     std::string model_path, float conf_thresh,
                                     int nms_dist) {
  CHECK_GT(width, 0);
  CHECK_GT(height, 0);
  CHECK_GT(max_number_of_features, 0);
  width_ = width;
  height_ = height;
  max_number_of_features_ = max_number_of_features;
  model_path_ = model_path;
  conf_thresh_ = conf_thresh;
  nms_dist_ = nms_dist;

  image_.create(height_, width_, CV_8UC1);

  try {
    // TODO(b51): Make shared_ptr normal useable
    module_ = torch::jit::load(model_path.c_str());
    // Load module on tx2
    // module_ = std::make_shared<torch::jit::script::Module>(
    //    torch::jit::load(model_path.c_str()));
    module_->to(torch::Device(torch::kCUDA, FLAGS_CUDA));
  } catch (const c10::Error& e) {
    LOG(FATAL) << "Error loading the model from " << model_path;
  }
  VLOG(3) << "SuperPoint detector parameters:";
  VLOG(3) << "      max number of features: " << max_number_of_features_;
  VLOG(3) << "      model loaded: " << model_path;
  is_initialized_ = true;
}

void SuperPointFeatureDetector::NMSFast(
    const Eigen::Matrix3Xf& pts, int width, int height, int dist_thresh,
    Eigen::Matrix3Xf& filtered_pts) {
  Eigen::MatrixXf sorted_descend_pts;
  sorted_descend_pts.resize(pts.rows(), pts.cols());

  // Sort with heatmap value in descending order
  std::vector<Eigen::Vector3f> vec;
  for (int i = 0; i < pts.cols(); i++)
    vec.push_back(pts.col(i));
  std::sort(vec.begin(), vec.end(),
            [](Eigen::Vector3f const& v1, Eigen::Vector3f const& v2) {
              return v1[2] > v2[2];
            });
  for (int i = 0; i < pts.cols(); i++)
    sorted_descend_pts.col(i) = vec[i];

  // Pad the border of the grid, so that we can NMS points near the border.
  int pad = dist_thresh;
  int paded_width = width + 2 * pad;
  int paded_height = height + 2 * pad;
  Eigen::MatrixXf grid = Eigen::MatrixXf::Zero(paded_width, paded_height);
#pragma omp parallel for
  for (int i = 0; i < sorted_descend_pts.cols(); i++) {
    grid(std::round(sorted_descend_pts.col(i)[0] + pad),
         std::round(sorted_descend_pts.col(i)[1] + pad)) = 1;
  }

  // Iterate through points, highest to lowest conf, suppress neighborhood.
  int count = 0;
  int pad_2 = 2 * pad;
  filtered_pts = Eigen::Matrix3Xf::Zero(3, sorted_descend_pts.cols());
  for (int i = 0; i < sorted_descend_pts.cols(); i++) {
    // Account for top and left padding.
    int x = std::round(sorted_descend_pts.col(i)[0] + pad);
    int y = std::round(sorted_descend_pts.col(i)[1] + pad);
    if (grid(x, y) == 1) {
      // Block of size(p, q), starting at (i, j)
      // dynamic-size block expression: matrix.block(i, j, p, q)
      // https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
      grid.block(x - pad, y - pad, pad_2 + 1, pad_2 + 1) =
          Eigen::MatrixXf::Zero(pad_2 + 1, pad_2 + 1);
      grid(x, y) = -1;
      // Ignore points along border
      if ((x - pad) < pad or (x - pad >= width - pad) or
          (y - pad) < pad or (y - pad >= height - pad))
        continue;
      filtered_pts.col(count++) =
          Eigen::Vector3f(x - pad, y - pad, sorted_descend_pts.col(i)[2]);
    }
  }
  // resize the matrix to rows x cols while leaving old values untouched
  filtered_pts.conservativeResize(Eigen::NoChange, count);
}

void SuperPointFeatureDetector::Detect(const cv::Mat& image,
                                       Eigen::Matrix3Xf& filtered_pts,
                                       torch::Tensor& descriptors) {
  int H = image.rows;
  int W = image.cols;
  // convert image to float in range [0.0, 1.0]
  std::vector<torch::jit::IValue> inputs;

  // convert cv mat to tensor and transport to cuda
  at::Tensor image_tensor =
      torch::from_blob(image.data, {1, 1, H, W}, at::kFloat);
  inputs.emplace_back(image_tensor.to(torch::Device(torch::kCUDA, FLAGS_CUDA)));

  // inference
  auto outputs = module_->forward(inputs).toTuple();
  torch::Tensor semi = outputs->elements()[0].toTensor();
  torch::Tensor coarse_desc = outputs->elements()[1].toTensor();

  // Deal with feature points locations
  semi = semi.squeeze();
  torch::Tensor dense = semi.exp();
  dense = dense / (dense.sum(0) + 0.00001);
  torch::Tensor nodust = dense.slice(0, 0, dense.size(0) - 1);
  int Hc = H / CELL;
  int Wc = W / CELL;
  nodust = nodust.transpose(0, 1).transpose(1, 2);
  torch::Tensor heatmap = nodust.reshape({Hc, Wc, CELL, CELL});
  heatmap = heatmap.transpose(1, 2);
  heatmap = heatmap.reshape({Hc * CELL, Wc * CELL});
  torch::Tensor heatmap_cpu = heatmap.to(torch::kCPU, /*non_blocking=*/true);

  // Eigen::Map<Eigen::MatrixXf> eig_heatmap(heatmap_cpu.data<float>(),
  //                                         heatmap.size(0), heatmap.size(1));
  cv::Mat mat_heatmap(heatmap_cpu.size(0), heatmap_cpu.size(1), CV_32F,
                      heatmap_cpu.data<float>());
  /**
   *  Filtered to binary mat first and than get non zero value coordinates
   *  cv::threshold(src, dst, threshold, max_binary_value, threshold_type)
   *  threshold_type: 0, Binary
   *                  1, Binary Inverted
   *                  2, Threshold Truncated
   *                  3, Threshold to Zero
   *                  4, Threshold to Zero Inverted
   */
  cv::Mat bin_mat;
  cv::threshold(mat_heatmap, bin_mat, conf_thresh_, 1, 0);
  bin_mat.convertTo(bin_mat, CV_8UC1);
  cv::Mat coordinates;
  cv::findNonZero(bin_mat, coordinates);

  Eigen::Matrix3Xf pts;
  pts.resize(3, coordinates.total());
#pragma omp parallel for
  for (size_t i = 0; i < coordinates.total(); i++) {
    pts.block<3, 1>(0, i) = Eigen::Vector3f(
        coordinates.at<cv::Point>(i).x, coordinates.at<cv::Point>(i).y,
        mat_heatmap.at<float>(coordinates.at<cv::Point>(i)));
  }
  // nms fast
  NMSFast(pts, W, H, nms_dist_, filtered_pts);

  VLOG(4) << "filtered points size: " << filtered_pts.cols();
  int number_of_features = filtered_pts.cols();

  // Process descriptor
  // Interpolate into descriptor map using 2D point locations.
  cv::Mat mat_measurements(2, number_of_features, CV_32F);
  for (int i = 0; i < number_of_features; i++) {
    // grid_sampler needs grid range between [-1., 1.]
    mat_measurements.at<float>(0, i) =
        (filtered_pts.col(i)[0] / (float(W) / 2.)) - 1.;
    mat_measurements.at<float>(1, i) =
        (filtered_pts.col(i)[1] / (float(H) / 2.)) - 1.;
    /*  For debug display
    Eigen::Vector3f p = filtered_pts.col(i);
    LOG(INFO) << "x: " << p[0] << " y: " << p[1] << " value: " << p[2];
    cv::circle(image, cv::Point(p[0], p[1]), 2, cv::Scalar(0, 255, 255));

    cv::imwrite("key_points.jpg", image);
    exit(0);
    */
  }
  torch::Tensor sample_pts =
      torch::from_blob(mat_measurements.data, {2, number_of_features},
                       at::kFloat)
          .to(torch::Device(torch::kCUDA, FLAGS_CUDA));
  sample_pts = sample_pts.transpose(0, 1);
  sample_pts = sample_pts.view({1, 1, -1, 2});

  /**
   *  grid_sampler(at::Tensor input, at::Tensor grid, interpolation_mode,
   *  padding_mode)
   *  grid: range needs to be [-1., 1]
   *  interpolation_mode:
   *      at::native::detail
   *      enum class GridSamplerInterpolation {Bilinear, Nearest};
   *  padding_mode
   *      at::native::detail
   *      enum class GridSamplerPadding {Zeros, Border, Reflection}
   */
  descriptors = at::grid_sampler(
      coarse_desc, sample_pts.to(torch::Device(torch::kCUDA, FLAGS_CUDA)), 0,
      0);
  descriptors = descriptors.reshape({coarse_desc.size(1), -1});
  descriptors = descriptors / (descriptors.norm(c10::nullopt, 0).view({1, -1}));
  descriptors = descriptors.to(torch::kCPU, true);
}

void SuperPointFeatureDetector::Detect(
    const cv::Mat& image, Eigen::Matrix2Xf* current_measurements,
    FeatureDescriptorf* current_feature_descriptors) {
  image.copyTo(image_);
  cv::Mat input_image;
  double H_scale = 1.;
  double W_scale = 1.;
  // resize to net input

  cv::resize(image, input_image, cv::Size(FLAGS_W, FLAGS_H), 0, 0,
             cv::INTER_AREA);
  H_scale = image.rows / FLAGS_H;
  W_scale = image.cols / FLAGS_W;

  // Superpoint net only accept gray image
  if (input_image.channels() > 1) {
    cv::cvtColor(input_image, input_image, CV_BGR2GRAY);
  }
  // normalize to [0.0, 1.0]
  cv::Mat normed_image;
  input_image.convertTo(normed_image, CV_32F, 1.0 / 255, 0);

  Eigen::Matrix3Xf filtered_pts;
  torch::Tensor descriptors;
  Detect(normed_image, filtered_pts, descriptors);
  // LOG(INFO) << " descriptor 0: " << descriptors.slice(1, 0, 1);

  int number_of_features = filtered_pts.cols();
  int descriptor_size = descriptors.size(0);
  current_measurements->resize(Eigen::NoChange, number_of_features);
  current_measurements->block(0, 0, 2, number_of_features) =
      filtered_pts.block(0, 0, 2, number_of_features);

  // restor to original size
  current_measurements->row(0) = current_measurements->row(0) * W_scale;
  current_measurements->row(1) = current_measurements->row(1) * H_scale;

  CHECK_EQ(number_of_features, descriptors.size(1));

  cv::Mat mat_descriptors(descriptors.size(0), descriptors.size(1), CV_32F,
                          descriptors.data<float>());
  // Make mat_descriptors width = descriptor_size (256 here)
  //                      height = number_features
  mat_descriptors = mat_descriptors.t();

  current_feature_descriptors->Configure(descriptor_size,
                                         number_of_features);
  current_feature_descriptors->Resize(number_of_features);
  if (mat_descriptors.isContinuous()) {
    // if continuous
    memcpy(current_feature_descriptors->descriptor(0), mat_descriptors.data,
           descriptor_size * number_of_features * sizeof(float));
  } else {
    for (int i = 0; i < number_of_features; ++i) {
      memcpy(current_feature_descriptors->descriptor(i),
             &mat_descriptors.at<float>(i, 0), descriptor_size * sizeof(float));
    }
  }
  // LOG(INFO) << current_feature_descriptors->Size();
  // LOG(INFO) << current_feature_descriptors->DescriptorSize();
  // Eigen::Map<Eigen::Matrix<float, 1, 256> > for_display(
  //     current_feature_descriptors->descriptor(0), 1, 256);
  // LOG(INFO) << for_display;
}
