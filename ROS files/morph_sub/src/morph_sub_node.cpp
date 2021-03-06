#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/rgb_colors.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include<vector>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>

int main(int argc, char** argv)
{
   ros::init(argc, argv, "morph_sub");
   ros::NodeHandle nh;
   image_transport::ImageTransport it(nh);
   image_transport::Publisher pub = it.advertise("camera/image", 1);
   cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
   cv::waitKey(30);
   sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

   ros::Rate loop_rate(5);
   while (nh.ok()) {
     pub.publish(msg);
     ros::spinOnce();
     loop_rate.sleep();
   }
}
