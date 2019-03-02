#include "src/qr.h"

#include <iostream>
#include <string>
#include <assert.h>

int main(int argc, char* argv[]) {
  
  if(argc < 2){
    std::cout << "Error: please provide video file" << std::endl;
    return 1;
  }
  
  cv::VideoCapture vc(/*argv[1]*/ 1);
  
  cv::Mat m;

  std::vector<uint16_t> found;

  std::cout << "0:" << std::endl;

  int i = 1;
  
  while(true){
    
    if(!vc.read(m)) break;
    
    i++;

    if(i < 50) continue;

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
      
      std::stringstream ss;
      ss << (i - 1) << ".png";
      cv::imwrite(ss.str(), im[0]);
      std::cout << i++ << ":" << std::endl;
    }

    cv::imshow("x", k);
    
    // std::cout << i << std::endl;
    
    cv::waitKey(30);
  }
  
  return 0;
}
