#pragma once

#include <opencv2/opencv.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

class OccupancyGridParam {
public:
    int height;
    int width;
    double resolution;
    double x;
    double y;
    double theta;
    cv::Mat R;
    cv::Mat t;

    void GetOccupancyGridParam(const nav_msgs::msg::OccupancyGrid &OccGrid);
    void Image2MapTransform(cv::Point &src_point, cv::Point2d &dst_point);
    void Map2ImageTransform(cv::Point2d &src_point, cv::Point &dst_point);
};
