#include <opencv/cv.hpp>
#include <vector>
#include <tuple>
#include <iostream>

int main() {

  cv::Mat m, g;

  cv::VideoCapture v(0);

  while (true) {

    v >> m;

    // cv::cvtColor(m, g, CV_BGR2GRAY);

    cv::inRange(m, cv::Scalar(50, 50, 50), cv::Scalar(255, 255, 255), g);

    cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9),
                                            cv::Point(3, 3));

    cv::dilate(g, g, element);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(g, contours, hierarchy, cv::RETR_TREE,
                     CV_CHAIN_APPROX_TC89_L1, cv::Point(0, 0));

    cv::Mat drawing = cv::Mat::zeros(g.size(), CV_8UC1);
    for (int i = 0; i < contours.size(); i++) {
      cv::Scalar color = cv::Scalar(255, 255, 255);
      cv::drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0,
                       cv::Point());
    }
    
    std::vector<cv::Vec2f> lines; // will hold the results of the detection
    cv::HoughLines(drawing, lines, 1, CV_PI / 90, 90, 0,
                   0); // runs the actual detection
    
    std::vector<cv::Point> intersections;
    
    std::vector<cv::Vec2f> lines_trimmed;
    
    for(size_t i = 0; i < lines.size(); i++){
      float rho = lines[i][0], theta = lines[i][1];
      double a = cos(theta), b = sin(theta);
      double x0 = a * rho, y0 = b * rho;
      bool k = false;
      for(size_t j = 0; j < lines_trimmed.size(); j++){
        float rhok = lines_trimmed[j][0], thetak = lines_trimmed[j][1];
        double a = cos(thetak), b = sin(thetak);
        double x0k = a * rhok, y0k = b * rhok;
        double dist = (x0k - x0)*(x0k - x0) + (y0k - y0)*(y0k - y0);
        if(dist < 5000 && (abs(theta - thetak) < 2 || abs(theta - thetak) > 5)) {
          k = true;
          break;
        }
      }
      if(k) continue;
      lines_trimmed.push_back(lines[i]);
    }
    
    // Draw the lines
    for (size_t i = 0; i < lines_trimmed.size(); i++) {
      for(size_t j = i + 1; j < lines_trimmed.size(); j++){
        auto standard = [&](int m){
          float rho = lines_trimmed[m][0], theta = lines_trimmed[m][1];
          double a = cos(theta), b = sin(theta);
          double x0 = a * rho, y0 = b * rho;
          double x1 = x0 + 1000 * (-b);
          double y1 = y0 + 1000 * (a);
          double x2 = x0 - 1000 * (-b);
          double y2 = y0 - 1000 * (a);
          return std::tuple<double, double, double>(y1 - y2, x2 - x1, (x1 - x2) * y1 + (y2 - y1) * x1);
        };
        auto [a1, b1, c1] = standard(i);
        auto [a2, b2, c2] = standard(j);
        double det = a1 * b2 - a2 * b1;
        if(det == 0) continue; // parallel lines
        double x = -(c1 * b2 - c2 * b1) / det;
        double y = -(a1 * c2 - a2 * c1) / det;
        std::cout << x << "," << y << std::endl;
        cv::circle(m, cv::Point(x, y), 3, cv::Scalar(0, 128, 255), 3);
      }
    }

    cv::imshow("M", m);
    int k;
    if ((k = cv::waitKey(1)) != 255) {
      printf("%i\n", k);
      break;
    }
  }

  return 0;
}




