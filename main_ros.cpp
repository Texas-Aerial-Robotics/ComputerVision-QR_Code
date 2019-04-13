#include "src/qr.h"

#include <assert.h>
#include <iostream>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt32.h>
#include <string>

int main(int argc, char *argv[]) {

  ros::init(argc, argv, "qrpart");
  ros::NodeHandle nh;
  ros::NodeHandle _nh("~");

  qr::qr_status status;

  std::string video_stream_provider;
  cv::VideoCapture vc;
  if (_nh.getParam("video_stream_provider", video_stream_provider)) {
    ROS_INFO_STREAM(
        "Resource video_stream_provider: " << video_stream_provider);
    // If we are given a string of 4 chars or less (I don't think we'll have
    // more than 100 video devices connected) treat is as a number and act
    // accordingly so we open up the videoNUMBER device
    if (video_stream_provider.size() < 4) {
      ROS_INFO_STREAM("Getting video from provider: /dev/video"
                      << video_stream_provider);
      vc.open(atoi(video_stream_provider.c_str()));
    } else {
      ROS_INFO_STREAM("Getting video from provider: " << video_stream_provider);
      std::cout << "Using compressed " << std::endl;
      vc.open(video_stream_provider);
      vc.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
      vc.set(CV_CAP_PROP_FRAME_WIDTH, 640);
      vc.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    }
  } else {
    ROS_ERROR("Failed to get param 'video_stream_provider'");
    return -1;
  }

  ros::Publisher pub = nh.advertise<std_msgs::String>("output", 5);
  ros::Publisher num = nh.advertise<std_msgs::UInt32>("", 1);

  cv::Mat m;

  std::vector<uint16_t> found;

  std::cout << "0:" << std::endl;

  int i = 1;

  ros::Rate r(30);

  while (true) {

    if (!vc.read(m))
      break;

    i++;

    if (i < 50)
      continue;

    while (pub.getNumSubscribers() == 0)
      ;

    cv::Mat k;

    m.copyTo(k);

    status.dirtyflag = false;

    localize(m, k, status);

    if (status.dirtyflag) {
      auto list = status.compute();
      if (list.size() != 0) {
        std::printf("----\n");
        for (int i = 0; i < list.size(); i++)
          std::printf("%i\n", list[i]);
      }
    }

    ros::spinOnce();
    r.sleep();
  }

  return 0;
}
