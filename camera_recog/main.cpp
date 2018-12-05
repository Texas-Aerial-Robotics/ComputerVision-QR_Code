#include <algorithm>
#include <iostream>
#include <opencv/cv.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include <chrono>

#include "qr.h"

std::vector<uint16_t> f;
int inum = 0;

void onQR(cv::Mat image, std::vector<uint16_t> possibilities) {
  inum++;
  for (int j = 0; j < possibilities.size(); j++) {
    auto x = possibilities[j];
    for (int i = 0; i < f.size(); i++) {
      if (f[i] == x)
        x = 0;
    }
    if (x != 0) {
      std::stringstream ss;
      ss << x;
      printf("%i\n", x);
      f.push_back(x);
      cv::imwrite("_" + ss.str() + ".png", image);
    }
  }
  /*
  std::stringstream ss;
  ss << inum;
  cv::putText(image, ss.str(), cv::Point(20,
  image.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 1.0,
  cv::Scalar(255, 0, 0));
              */
  cv::imshow("onqr", image);
  cv::imwrite("dr.png", image);
}

int main() {

  cv::Mat m, g;

  cv::VideoCapture v(0);

  auto starttime = std::chrono::high_resolution_clock::now();

  while (true) {

    if (!(v.read(m))) {
      cv::waitKey(0);
      break;
    }

    double difftime =
        (std::chrono::high_resolution_clock::now() - starttime).count();

    // cv::cvtColor(m, g, CV_BGR2GRAY);

    cv::inRange(
        m, cv::Scalar(50, 10 + 10 * sin(difftime), 10 + 10 * sin(difftime)),
        cv::Scalar(255, 255, 255), g);

    cv::Mat element =
        getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9), cv::Point(3, 3));

    //cv::erode(g, g, element);

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
      approxPolyDP(cv::Mat(contours[i]), contours_poly, 30, true);
      if (contours_poly.size() == 4) {
        cv::circle(drawing, cv::Point(contours_poly[0].x, contours_poly[0].y),
                   5, cv::Scalar(255, 255, 255), 5);
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
        float range[] = {0, 256}; // the upper boundary is
                                  // exclusive
        const float *histRange = {range};
        bool uniform = true, accumulate = false;
        cv::Mat hist;
        cv::calcHist(&transformed, 1, 0, cv::Mat(), hist, 1, &histSize,
                     &histRange, uniform, accumulate);
        
        cv::Mat bound_eroded(transformed);

        // cv::dilate(transformed, transformed, element); 
        cv::erode(bound_eroded,
         bound_eroded, element);
        cv::erode(bound_eroded, bound_eroded,
         element);

        std::stringstream ss;
        ss << "i" << (image_i++);

        std::vector<std::vector<cv::Point>> innercontours;
        std::vector<cv::Vec4i> innerhierarchy;

        cv::findContours(bound_eroded, innercontours, innerhierarchy,
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

        // cv::Mat tfd = transformed(cv::Rect(x1, y1, x2 - x1, y2 - y1));

        // cv::imshow("tfd", tfd);

        double score = fabs(1 - (double)(x2 - x1) / (double)(y2 - y1));
        std::stringstream s2;

        if (/*score > 0.5 || */ score != score)
          continue; // score != score is true iff
                    // score is NaN

        cv::Mat test(transformed.size(), CV_8UC1);

        cv::Canny(transformed, test, 50, 200, 3);

        std::vector<cv::Vec2f> lines;
        cv::HoughLines(test, lines, 3, CV_PI / 180, 50);

        bool isortho = true;

        /*
        for (size_t i = 0; i < lines.size(); i++) {
          float rho = lines[i][0], theta = lines[i][1];
          cv::Point pt1, pt2;
          double a = cos(theta), b = sin(theta);
          double x0 = a * rho, y0 = b * rho;
          pt1.y = cvRound(y0 + 1000 * (a));
          pt2.y = cvRound(y0 - 1000 * (a));
          double dx = (x0 - b) / (x0 + b);
          double dy = (y0 + a) / (y0 - a);
          if(dx / dy > 0.3 || dy / dx > 0.3){
            isortho = false;
            break;
          }
        }
        if (!isortho)
          continue;
*/
        double stride_c = (x2 - x1) / 11.0;
        double stride_r = (y2 - y1) / 11.0;

        qr_comb_t qr(11);
        
        /*
        printf("-----------\n");
        for (int r = 0; r < 11; r++) {
          for (int c = 0; c < 11; c++) {
            int x = std::max((int)(x1 + stride_c * c), transformed.cols - 1);
            int y = std::max((int)(y1 + stride_r * r), transformed.rows - 1);
            int w = std::min(transformed.cols - 1 - (int)(x1 + stride_c * c), (int) stride_c);
            int h = std::min(transformed.rows - 1 - (int)(x1 + stride_r * r), (int) stride_r);
            auto dr = cv::Rect(x, y, w, h);
            printf("Rect: %i, %i, %i, %i\n", x, y, w, h);
            cv::Mat rec = transformed(dr);
            cv::circle(drawing, cv::Point(x, y), 5, cv::Scalar(255,255,255), 5);
            printf("%u\t", transformed.at<uchar>(x, y));
            qr(c, r) = transformed.at<uchar>(y, x) > 4 ? white : black;
            //printf("%s", qr(c, r) ? "#" : " ");
          }
          printf("\n");
        }
        */
        
        // printf("-----------\n");
        for (int r = 0; r < 11; r++) {
          for (int c = 0; c < 11; c++) {
            int stridew =
                transformed.cols - 1 -
                (x1 + (int)(stride_c * c));
            int strideh =
                transformed.rows - 1 -
                (y1 + (int)(stride_r * r));
            int x = x1 + (int)(stride_c * c);
            int y = y1 + (int)(stride_r * r);
            int w =
                std::min((int)stride_c, stridew);
            int h =
                std::min((int)stride_r, strideh);
            cv::Rect pixel(x, y, w, h);
            double avg = 0;
            int numm = 0;
            for (double d = 0.1; d < 1.0;
                 d += 0.2)
              for (double e = 0.1; e < 1.0;
                   e += 0.2) {
                numm++;
                auto pt = cv::Point(x + w * d,
                                    y + h * e);
                cv::circle(drawingn, pt, 1,
                           cv::Scalar(0, 0, 255),
                           1);
                uchar colour =
                    transformed.at<uchar>(pt);
                avg += colour;
              }
            avg /= numm;
            // printf("%.2f\t", avg);
            // if(fabs(avg - 128)) printf("wierd
            // code: %f\n", fabs(avg - 128));
            qr(c, r) = avg > 127 ? white : black;

            // printf("%s", avg > 128 ? " " :
            // "#");
          }
          // printf("\n");
}
        
        cv::Mat k = transformed(cv::Rect(x1, y1, x2 - x1, y2 - y1));

        std::vector<uint16_t> values;

        qr.compute(values);

        if(!values.empty())
          onQR(k, values);

        //cv::imshow("drawing", drawingn);
        //cv::imwrite("drawingn.png", drawingn);
      }
    }

    cv::imshow("M", drawing);
    int k;
    if ((k = cv::waitKey(1)) != 255) {
      printf("%i\n", k);
      break;
    }
  }

  return 0;
}
