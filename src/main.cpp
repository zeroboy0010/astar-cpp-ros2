#include <iostream>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <opencv2/opencv.hpp>
#include "astar_pkg/Astar.h"
#include "astar_pkg/OccMapTransform.h"

using namespace cv;
using namespace std;

class AstarNode : public rclcpp::Node
{
public:
    AstarNode() :   Node("astar"),
                    map_flag(false), 
                    startpoint_flag(false), 
                    targetpoint_flag(false), 
                    start_flag(false)
    {
        this->declare_parameter<bool>("Euclidean", true);
        this->declare_parameter<int>("OccupyThresh", -1);
        this->declare_parameter<double>("InflateRadius", 0.5);
        this->declare_parameter<int>("rate", 10);

        this->get_parameter("Euclidean", config.Euclidean);
        this->get_parameter("OccupyThresh", config.OccupyThresh);
        this->get_parameter("InflateRadius", InflateRadius);
        this->get_parameter("rate", rate);

        map_sub = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "map", 10, std::bind(&AstarNode::MapCallback, this, std::placeholders::_1));
        startPoint_sub = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "initialpose", 10, std::bind(&AstarNode::StartPointCallback, this, std::placeholders::_1));
        targetPoint_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "goal_pose", 10, std::bind(&AstarNode::TargetPointCallback, this, std::placeholders::_1));

        mask_pub = this->create_publisher<nav_msgs::msg::OccupancyGrid>("mask", 1);
        path_pub = this->create_publisher<nav_msgs::msg::Path>("nav_path", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / rate), std::bind(&AstarNode::TimerCallback, this));
    }

private:
    void MapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {   
        RCLCPP_INFO(this->get_logger(), "Received a map message");
        OccGridParam.GetOccupancyGridParam(*msg);

        int height = OccGridParam.height;
        int width = OccGridParam.width;
        int OccProb;
        Mat Map(height, width, CV_8UC1);
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                OccProb = msg->data[i * width + j];
                OccProb = (OccProb < 0) ? 100 : OccProb;
                Map.at<uchar>(height - i - 1, j) = 255 - round(OccProb * 255.0 / 100.0);
            }
        }

        Mat Mask;
        config.InflateRadius = round(InflateRadius / OccGridParam.resolution);
        astar.InitAstar(Map, Mask, config);

        OccGridMask.header.stamp = this->now();
        OccGridMask.header.frame_id = "odom";
        OccGridMask.info = msg->info;
        OccGridMask.data.clear();
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                OccProb = Mask.at<uchar>(height - i - 1, j) * 255;
                OccGridMask.data.push_back(OccProb);
            }
        }

        map_flag = true;
        startpoint_flag = false;
        targetpoint_flag = false;
    }

    void StartPointCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received a start point message");
        Point2d src_point = Point2d(msg->pose.pose.position.x, msg->pose.pose.position.y);
        OccGridParam.Map2ImageTransform(src_point, startPoint);

        startpoint_flag = true;
        if (map_flag && startpoint_flag && targetpoint_flag)
        {
            start_flag = true;
        }
    }

    void TargetPointCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received a target point message");
        Point2d src_point = Point2d(msg->pose.position.x, msg->pose.position.y);
        OccGridParam.Map2ImageTransform(src_point, targetPoint);

        targetpoint_flag = true;
        if (map_flag && startpoint_flag && targetpoint_flag)
        {
            start_flag = true;
        }
    }

    void TimerCallback()
    {
        if (start_flag)
        {
            double start_time = this->now().seconds();
            vector<Point> PathList;
            astar.PathPlanning(startPoint, targetPoint, PathList);
            if (!PathList.empty())
            {
                path.header.stamp = this->now();
                path.header.frame_id = "odom";
                path.poses.clear();
                for (size_t i = 0; i < PathList.size(); i++)
                {
                    Point2d dst_point;
                    OccGridParam.Image2MapTransform(PathList[i], dst_point);

                    geometry_msgs::msg::PoseStamped pose_stamped;
                    pose_stamped.header.stamp = this->now();
                    pose_stamped.header.frame_id = "map";
                    pose_stamped.pose.position.x = dst_point.x;
                    pose_stamped.pose.position.y = dst_point.y;
                    pose_stamped.pose.position.z = 0;
                    path.poses.push_back(pose_stamped);
                }
                path_pub->publish(path);
                double end_time = this->now().seconds();

                RCLCPP_INFO(this->get_logger(), "Find a valid path successfully! Use %f s", end_time - start_time);
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "Can not find a valid path");
            }

            start_flag = false;
        }

        if (map_flag)
        {
            mask_pub->publish(OccGridMask);
        }
    }

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr startPoint_sub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr targetPoint_sub;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr mask_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
    rclcpp::TimerBase::SharedPtr timer_;

    nav_msgs::msg::OccupancyGrid OccGridMask;
    nav_msgs::msg::Path path;
    pathplanning::AstarConfig config;
    pathplanning::Astar astar;
    OccupancyGridParam OccGridParam;
    Point startPoint, targetPoint;

    double InflateRadius;
    bool map_flag;
    bool startpoint_flag;
    bool targetpoint_flag;
    bool start_flag;
    int rate;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AstarNode>());
    rclcpp::shutdown();
    return 0;
}
