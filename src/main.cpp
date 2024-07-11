#include <iostream>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <opencv2/opencv.hpp>
#include <tf2/LinearMath/Quaternion.h> // Add this line
#include <geometry_msgs/msg/quaternion.hpp>

#include "astar_pkg/Astar.h"
#include "astar_pkg/OccMapTransform.h"

using namespace cv;
using namespace std;

double quaternionToYaw(double x, double y, double z, double w)
{
    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    return std::atan2(siny_cosp, cosy_cosp);
}


class AstarNode : public rclcpp::Node
{
public:
    AstarNode() : Node("astar"), map_flag(false), startpoint_flag(false), targetpoint_flag(false), start_flag(false)
    {
        this->declare_parameter<bool>("Euclidean", true);
        this->declare_parameter<int>("OccupyThresh", -1);
        this->declare_parameter<double>("InflateRadius", 0.2);
        this->declare_parameter<int>("rate", 10);

        this->get_parameter("Euclidean", config.Euclidean);
        this->get_parameter("OccupyThresh", config.OccupyThresh);
        this->get_parameter("InflateRadius", InflateRadius);
        this->get_parameter("rate", rate);

        map_sub = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "map", 10, std::bind(&AstarNode::MapCallback, this, std::placeholders::_1));
        targetPoint_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10, std::bind(&AstarNode::TargetPointCallback, this, std::placeholders::_1));
        lidar_sub = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10, std::bind(&AstarNode::LidarCallback, this, std::placeholders::_1));
        amcl_pose_sub = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "amcl_pose", 10, std::bind(&AstarNode::AmclPoseCallback, this, std::placeholders::_1));

        mask_pub = this->create_publisher<nav_msgs::msg::OccupancyGrid>("mask", 1);
        path_pub = this->create_publisher<nav_msgs::msg::Path>("nav_path", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / rate), std::bind(&AstarNode::TimerCallback, this));
    }

private:
    void MapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        OccGridParam.GetOccupancyGridParam(*msg);

        int height = OccGridParam.height;
        int width = OccGridParam.width;
        int OccProb;
        Map = Mat(height, width, CV_8UC1);
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                OccProb = msg->data[i * width + j];
                OccProb = (OccProb < 0) ? 100 : OccProb;
                Map.at<uchar>(height - i - 1, j) = 255 - round(OccProb * 255.0 / 100.0);
            }
        }

        map_flag = true;
        startpoint_flag = false;
        targetpoint_flag = false;
    }

    void TargetPointCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        Point2d src_point = Point2d(msg->pose.position.x, msg->pose.position.y);
        OccGridParam.Map2ImageTransform(src_point, targetPoint);

        targetpoint_flag = true;
        if (map_flag && startpoint_flag && targetpoint_flag)
        {
            start_flag = true;
        }
    }

    void LidarCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        if (!map_flag)
        {
            return;
        }

        int height = OccGridParam.height;
        int width = OccGridParam.width;
        Mat lidarMap = Map.clone();

        double angle = msg->angle_min;
        for (auto range : msg->ranges)
        {
            if (range < msg->range_max && range > msg->range_min)
            {
                double x = range * cos(angle + robot_theta) + robot_x;
                double y = range * sin(angle + robot_theta) + robot_y;


                Point2d map_point(x, y);
                Point img_point;
                OccGridParam.Map2ImageTransform(map_point, img_point);

                if (img_point.x >= 0 && img_point.x < width && img_point.y >= 0 && img_point.y < height)
                {
                    lidarMap.at<uchar>(img_point.y, img_point.x) = 0; // Mark LiDAR obstacles in the lidarMap
                }
            }
            angle += msg->angle_increment;
        }

        // Mat Mask;
        config.InflateRadius = round(InflateRadius / OccGridParam.resolution);
        astar.InitAstar(lidarMap, Mask, config);

        OccGridMask.header.stamp = this->now();
        OccGridMask.header.frame_id = "map";
        OccGridMask.info.height = height;
        OccGridMask.info.width = width;
        OccGridMask.info.resolution = OccGridParam.resolution;
        OccGridMask.info.origin.position.x = OccGridParam.x;
        OccGridMask.info.origin.position.y = OccGridParam.y;
        OccGridMask.info.origin.position.z = 0.0;
        OccGridMask.info.origin.orientation.w = 1.0;
        OccGridMask.data.clear();
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int occ_prob = Mask.at<uchar>(height - i - 1, j) * 255;
                OccGridMask.data.push_back(occ_prob);
            }
        }

        mask_pub->publish(OccGridMask);
    }

    void AmclPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        robot_x = msg->pose.pose.position.x;
        robot_y = msg->pose.pose.position.y;

        robot_theta = quaternionToYaw(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);

        Point2d src_point = Point2d(msg->pose.pose.position.x, msg->pose.pose.position.y);
        OccGridParam.Map2ImageTransform(src_point, startPoint);

        startpoint_flag = true;
        if (map_flag && startpoint_flag && targetpoint_flag)
        {
            start_flag = true;
        }
    }

    void TimerCallback()
    {
        if (start_flag)
        {
            vector<Point> PathList;
            astar.PathPlanning(startPoint, targetPoint, PathList);

            if (!PathList.empty())
            {
                // Check for obstacles along the path
                bool obstacle_detected = CheckPathForObstacles(PathList);

                if (obstacle_detected)
                {
                    RCLCPP_WARN(this->get_logger(), "Local obstacle detected, replanning path...");

                    // Re-plan path with updated obstacle information
                    Mat updatedMap = Map.clone();
                    astar.InitAstar(updatedMap, Mask, config);
                    astar.PathPlanning(startPoint, targetPoint, PathList);
                }

                path.header.stamp = this->now();
                path.header.frame_id = "map";
                path.poses.clear();
                for (size_t i = 0; i < PathList.size(); i++)
                {
                    Point2d dst_point;
                    OccGridParam.Image2MapTransform(PathList[i], dst_point);
                    geometry_msgs::msg::PoseStamped pose_stamped;
                    pose_stamped.pose.position.x = dst_point.x;
                    pose_stamped.pose.position.y = dst_point.y;
                    pose_stamped.pose.position.z = 0.0;
                    path.poses.push_back(pose_stamped);
                }

                RCLCPP_INFO(this->get_logger(), "Path updated successfully");
                path_pub->publish(path);
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "No path found!");
            }

            if (lidar_map_updated)
            {
                // If there was an update from the LiDAR data, recompute the path
                start_flag = true;
                lidar_map_updated = false;
            }
            else
            {
                start_flag = false;
            }
        }
    }

    bool CheckPathForObstacles(const vector<Point>& path)
    {
        int height = OccGridParam.height;
        int width = OccGridParam.width;
        Mat pathMap = Map.clone();

        for (const auto& point : path)
        {
            Point src_point(point.x, point.y);
            Point2d map_point;
            OccGridParam.Image2MapTransform(src_point, map_point);

            if (map_point.x >= 0 && map_point.x < width && map_point.y >= 0 && map_point.y < height)
            {
                if (pathMap.at<uchar>(map_point.y, map_point.x) == 0)
                {
                    return true; // Obstacle found along the path
                }
            }
        }

        return false; // No obstacles found along the path
    }



    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr targetPoint_sub;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_pose_sub;

    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr mask_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;

    rclcpp::TimerBase::SharedPtr timer_;

    pathplanning::Astar astar;
    OccupancyGridParam OccGridParam;
    nav_msgs::msg::OccupancyGrid OccGridMask;
    nav_msgs::msg::Path path;

    Mat Map;
    Mat Mask;
    Point startPoint;
    Point targetPoint;

    bool map_flag;
    bool startpoint_flag;
    bool targetpoint_flag;
    bool start_flag;
    bool lidar_map_updated = false;

    double robot_x, robot_y, robot_theta;

    pathplanning::AstarConfig config;
    double InflateRadius;
    int rate;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AstarNode>());
    rclcpp::shutdown();
    return 0;
}
