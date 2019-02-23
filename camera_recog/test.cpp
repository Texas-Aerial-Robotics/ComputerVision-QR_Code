#include "src/qr.h"

#include <iostream>
#include <string>
#include <assert.h>

int main(int argc, char* argv[]) {
  
  if(argc < 2){
    std::cout << "Error: please provide video file" << std::endl;
    return 1;
  }
  
  cv::VideoCapture vc(argv[1]);
  
  cv::Mat m;
  
  std::vector<uint16_t> found;

  std::cout << "0:" << std::endl;

  int i = 1;
  
  while(true){
    
    if(!vc.read(m)) break;
    
    cv::Mat k;
    
    m.copyTo(k);
    
    auto [qr, im] = localize(m, k);

    bool dr = false;

    for(int i = 0; i < qr.size(); i++){
      [&](){
        for(int j = 0; j < found.size(); j++){
          if(qr[i] == found[j]) return;
        }
        std::cout << qr[i] << std::endl;
        found.push_back(qr[i]);
        dr = true;
      }();
    }

    if(dr){
      std::stringstream ss;
      ss << (i - 1);
      cv::imshow(ss.str(), im[0]);
      std::cout << i++ << ":" << std::endl;
    }

    // cv::imshow("x", k);
    
    // std::cout << i << std::endl;
    
    cv::waitKey(1);
  }
  
  return 0;
}