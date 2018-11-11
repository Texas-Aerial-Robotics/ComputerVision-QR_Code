#include <iostream>
#include <opencv/cv.hpp>
#include <tuple>
#include <vector>
#include <utility>

#include "qr.h"

void onQR(cv::Mat image, std::vector<uint16_t> possibilities){
  cv::imshow("onqr", image);
}

int main() {

  cv::Mat m, g;

  cv::VideoCapture v("testing_data.mkv");
  
  while (true) {

    if (!(v.read(m))) {
      cv::waitKey(0);
      break;
    }
    // cv::cvtColor(m, g, CV_BGR2GRAY);

    cv::inRange(m, cv::Scalar(50, 1, 1), cv::Scalar(255, 255, 255), g);

    cv::Mat element =
        getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9), cv::Point(3, 3));

    cv::dilate(g, g, element);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(g, contours, hierarchy, cv::RETR_TREE,
                     CV_CHAIN_APPROX_TC89_L1, cv::Point(0, 0));

    cv::Mat drawing = cv::Mat::zeros(g.size(), CV_8UC1);
    int image_i = 0;
    for (int i = 0; i < contours.size(); i++) {
      cv::Scalar color = cv::Scalar(255, 255, 255);
      cv::drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0,
                       cv::Point());

      std::vector<cv::Point> contours_poly;
      contours_poly.resize(contours[i].size());
      approxPolyDP(cv::Mat(contours[i]), contours_poly, 15, true);
      if (contours_poly.size() == 4) {
        std::vector<cv::Point2f> quad_pts;
        std::vector<cv::Point2f> squre_pts;
        quad_pts.push_back(cv::Point2f(contours_poly[0].x, contours_poly[0].y));
        quad_pts.push_back(cv::Point2f(contours_poly[1].x, contours_poly[1].y));
        quad_pts.push_back(cv::Point2f(contours_poly[3].x, contours_poly[3].y));
        quad_pts.push_back(cv::Point2f(contours_poly[2].x, contours_poly[2].y));
        squre_pts.push_back(cv::Point2f(0, 0));
        squre_pts.push_back(cv::Point2f(0, g.rows));
        squre_pts.push_back(cv::Point2f(g.cols, 0));
        squre_pts.push_back(cv::Point2f(g.cols, g.rows));

        cv::Mat transmtx = getPerspectiveTransform(quad_pts, squre_pts);
        cv::Mat transformed = cv::Mat::zeros(g.rows, g.cols, CV_8UC1);
        warpPerspective(g, transformed, transmtx, g.size());

        int histSize = 256;
        float range[] = {0, 256}; // the upper boundary is exclusive
        const float *histRange = {range};
        bool uniform = true, accumulate = false;
        cv::Mat hist;
        cv::calcHist(&transformed, 1, 0, cv::Mat(), hist, 1, &histSize,
                     &histRange, uniform, accumulate);
        float avg = 0.0;
        for (int j = 0; j < histSize; j++)
          avg +=
              j * hist.at<float>(j - 1) / transformed.rows / transformed.cols;
        if (avg < 2.0 || avg > 4.0)
          continue;

        cv::dilate(transformed, transformed, element);
        cv::erode(transformed, transformed, element);
        cv::erode(transformed, transformed, element);

        std::stringstream ss;
        ss << "i" << (image_i++);

        std::vector<std::vector<cv::Point>> innercontours;
        std::vector<cv::Vec4i> innerhierarchy;

        cv::findContours(transformed, innercontours, innerhierarchy,
                         cv::RETR_TREE, CV_CHAIN_APPROX_TC89_L1,
                         cv::Point(0, 0));

        cv::Mat drawingn = cv::Mat::zeros(transformed.size(), CV_8UC3);

        int x1 = -1, x2 = -1, y1 = -1, y2 = -1;

        for (int j = 0; j < innercontours.size(); j++) {
          cv::Scalar color = cv::Scalar(255, 255, 255);
          cv::drawContours(drawingn, innercontours, j, color, 2, 8, hierarchy,
                           0, cv::Point());
          cv::Rect boundRect = boundingRect(innercontours[j]);
          if (boundRect.x == 0 && boundRect.y == 0)
            continue;
          if (x1 == -1) {
            x1 = boundRect.x;
            y1 = boundRect.y;
            x2 = x1 + boundRect.width;
            y2 = y1 + boundRect.height;
          } else {
            x1 = (x1 < boundRect.x) ? x1 : boundRect.x;
            y1 = (y1 < boundRect.y) ? y1 : boundRect.y;
            int _x2 = boundRect.x + boundRect.width;
            int _y2 = boundRect.y + boundRect.height;
            x2 = (x2 > _x2) ? x2 : _x2;
            y2 = (y2 > _y2) ? y2 : _y2;
          }
        }

        double score = fabs(1 - (double)(x2 - x1) / (double)(y2 - y1));
        std::stringstream s2;

        if (score > 0.5 || score != score)
          continue; // score != score is true if score is NaN

        double stride = ((x2 - x1) > (y2 - y1) ? (x2 - x1) : (y2 - y1)) / 11.0;
        
        qr_comb_t qr(11);
        
        for (int c = 0; c < 11; c++) {
          for (int r = 0; r < 11; r++) {
            cv::Point p(x1 + (int)(stride * r + 0.5 * stride),
                        y1 + (int)(stride * c + 0.5 * stride));
            cv::circle(drawingn, p, 5, cv::Scalar(0, 0, 255), 5);
            auto v = transformed.at<uchar>(p);
            qr(r, c) = v > 128 ? white : black;
          }
        }
        
        std::vector<uint16_t> values;
        
        qr.compute(values);
        
        onQR(transformed, values);
      }
    }

    cv::imshow("M", m);
    int k;
    if ((k = cv::waitKey(1)) != 255) {
      printf("%i\n", k);
      break;
    }
  }
  
  fflush(stdout);

  return 0;
}
