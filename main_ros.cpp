#include "src/qr.h"

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <iostream>
#include <string>
#include <assert.h>

int main(int argc, char* argv[]) {

  ros::init(argc, argv, "qrpart");
  ros::NodeHandle nh;
  ros::NodeHandle _nh("~");

  std::string video_stream_provider;
  cv::VideoCapture vc;
  if (_nh.getParam("video_stream_provider", video_stream_provider)){
      ROS_INFO_STREAM("Resource video_stream_provider: " << video_stream_provider);
      // If we are given a string of 4 chars or less (I don't think we'll have more than 100 video devices connected)
      // treat is as a number and act accordingly so we open up the videoNUMBER device
      if (video_stream_provider.size() < 4){
          ROS_INFO_STREAM("Getting video from provider: /dev/video" << video_stream_provider);
          vc.open(atoi(video_stream_provider.c_str()));
      }
      else{
          ROS_INFO_STREAM("Getting video from provider: " << video_stream_provider);

          std::cout << "Using compressed " << std::endl;
          vc.open(video_stream_provider);
          vc.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') );
          vc.set(CV_CAP_PROP_FRAME_WIDTH ,640);
          vc.set(CV_CAP_PROP_FRAME_HEIGHT ,480);
      }
  }
  else{
      ROS_ERROR("Failed to get param 'video_stream_provider'");
      return -1;
  }

  ros::Publisher pub = nh.advertise<std_msgs::String>("output", 5);

  cv::Mat m;

  std::vector<uint16_t> found;

  std::cout << "0:" << std::endl;

  int i = 1;

  ros::Rate r(30);

  while(true){

    if(!vc.read(m)) break;

    i++;

    if(i < 50) continue;

    while(pub.getNumSubscribers() == 0) ;

    cv::Mat k;

    m.copyTo(k);

    auto [qr, im] = localize(m, k);

    bool dr = false;

    for(int i = 0; i < qr.size(); i++){
      [&](){
        for(int j = 0; j < found.size(); j++){
          if(qr[i] == found[j]) return;
        }
        //std::cout << qr[i] << std::endl;
        found.push_back(qr[i]);
        dr = true;
      }();
    }

    if(dr){
      std::set<int> n1, n2, n3, n4;
      for(int i = 0; i < qr.size(); i++){
        n1.insert((qr[i] / 1000) % 10);
        n2.insert((qr[i] / 100) % 10);
        n3.insert((qr[i] / 10) % 10);
        n4.insert(qr[i] % 10);
      }
#define show(n) for(auto i = n.begin(); i != n.end(); i++) std::cout << (i == n.begin() ? "" : ", ") << (int)(*i);
      std::cout << "[";
      show(n1)
      std::cout << "][";
      show(n2)
      std::cout << "][";
      show(n3)
      std::cout << "][";
      show(n4)
      std::cout << "]" << std::endl;
#undef show
      std::stringstream ss;
      ss << (i - 1) << ".png";
      cv::imwrite(ss.str(), im[0]);
      // ROS integration
      std_msgs::String msg;
      msg.data = ss.str();
      pub.publish(msg);
    }
    ros::spinOnce();
    r.sleep();
  }

  return 0;
}
