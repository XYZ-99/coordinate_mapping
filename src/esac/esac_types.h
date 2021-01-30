#pragma once

#include "opencv2/opencv.hpp"

namespace esac
{
  typedef std::pair<cv::Mat, cv::Mat> pose_t;
  typedef cv::Mat_<double> trans_t;
  typedef at::TensorAccessor<float, 4> coord_t;
  typedef at::TensorAccessor<long, 1> hyp_assign_t;
}
