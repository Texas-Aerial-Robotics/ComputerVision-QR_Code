#include "qr.h"

#include <cmath>
#include <opencv2/opencv.hpp>

#include <set>

struct mline {
  int x1, x2, y1, y2;
};

#define THRESH 25

template <typename T, typename T2> void dist(cv::Point pt, T &pts, T2 &areas) {
  for (int j = 0; j < pts.size(); j++) {
    if (pts[j].first)
      continue;
    if (cv::norm(pts[j].second - pt) < THRESH) {
      areas.back().push_back(pts[j].second);
      pts[j].first = true;
      dist(pts[j].second, pts, areas);
    }
  }
}

double dist(cv::Point p1, cv::Point p2){
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

void localize(cv::Mat &m, cv::Mat &output, qr::qr_status& status) {

  cv::Mat d, r;

  m.copyTo(d);

  cv::inRange(d, cv::Scalar(40, 40, 40), cv::Scalar(255, 255, 255), d);

  // blur(d, d, cv::Size(3, 3));

  cv::Canny(d, r, 20, 60, 3);

  //cv::imshow("r", r);

  std::vector<cv::Vec2f> lines;
  std::vector<std::pair<bool, cv::Point>> pts;
  std::vector<std::vector<cv::Point>> areas;

  cv::HoughLines(r, lines, 1, CV_PI / 60, 34, 0, 0);

  for (size_t i = 0; i < lines.size(); i++) {
    double r1 = lines[i][0], t1 = lines[i][1];
    for (size_t j = 0; j < lines.size(); j++) {
      if (i == j)
        continue;
      double r2 = lines[j][0], t2 = lines[j][1];

      double det = cos(t1) * sin(t2) - sin(t1) * cos(t2);
      if (det == 0)
        continue;

      double x = r1 / det * sin(t2) - r2 / det * sin(t1);
      double y = r2 / det * cos(t1) - r1 / det * cos(t2);

      // std::cout << x << ", " << y << std::endl;

      pts.emplace_back(false, cv::Point(x, y));
    }
  }
  /*
  if(pts.size() > 2){

  double avg = 0;
  for(int i = 0; i < pts.size(); i++){
    cv::Point cp = pts[i == 0 ? 1 : 0];
    for(int j = 1; j < pts.size(); j++){
      
    }
  }
  avg /= pts.size() * pts.size();

  std::cout << avg << std::endl;
  
  }
  */
  if(pts.size() > 5000) return;
  for (int i = 0; i < pts.size(); i++) {
    if (pts[i].first)
      continue;
    areas.push_back({pts[i].second});
    pts[i].first = true;
    dist(pts[i].second, pts, areas);
  }

  cv::RNG rng(122112);

  for (int i = 0; i < areas.size(); i++) {
    cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255),
                     rng.uniform(0, 255));
    auto rect = minAreaRect(areas[i]);
    for (int j = 0; j < areas[i].size(); j++) {
      cv::circle(output, areas[i][j], 2, color, -1);
    }
    if (rect.size.area() < 5)
      continue;
    cv::Point2f rect_points[4];
    rect.points(rect_points);

    std::vector<cv::Point2f> squre_pts;
    squre_pts.push_back(cv::Point2f(0, d.rows));
    squre_pts.push_back(cv::Point2f(0, 0));
    squre_pts.push_back(cv::Point2f(d.cols, 0));
    squre_pts.push_back(cv::Point2f(d.cols, d.rows));

    std::vector<cv::Point2f> rect_pts;
    for (int i = 0; i < 4; i++)
      rect_pts.push_back(rect_points[i]);

    cv::Mat transmtx = getPerspectiveTransform(rect_pts, squre_pts);
    cv::Mat transformed = cv::Mat::zeros(d.rows, d.cols, CV_8UC1);
    cv::Mat tfcopy;
    warpPerspective(d, transformed, transmtx, d.size());

    cv::cvtColor(transformed, tfcopy, CV_GRAY2BGR);

    double ratio = rect.size.width / (double)rect.size.height;

    if (ratio > 1.0)
      ratio = 1.0 / ratio;

    if (ratio < 0.745)
      continue;

    qr::qr_t q(11);

    for (int j = 0; j < 4; j++)
      line(output, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);

    for (int i = 0; i < 11; i++) {
      double ratio = transformed.rows / 11.0;
      int y1 = int(ratio * i);
      int y2 = int(ratio * (i + 1));
      for (int j = 0; j < 11; j++) {
        double ratio = transformed.cols / 11.0;
        int x1 = int(ratio * j);
        int x2 = int(ratio * (j + 1));
        line(tfcopy, cv::Point(x1, y1), cv::Point(x1, y2),
             cv::Scalar(255, 0, 0), 1, 8);
        line(tfcopy, cv::Point(x1, y2), cv::Point(x2, y2),
             cv::Scalar(255, 0, 0), 1, 8);
        line(tfcopy, cv::Point(x2, y2), cv::Point(x2, y1),
             cv::Scalar(255, 0, 0), 1, 8);
        line(tfcopy, cv::Point(x2, y1), cv::Point(x1, y1),
             cv::Scalar(255, 0, 0), 1, 8);
        cv::Rect r(x1, y1, x2 - x1, y2 - y1);
        cv::Mat sample = transformed(r);
        int count = 0;
        int threshold = sample.rows * sample.cols / 2;
        for (int i = 0; i < sample.rows; i++) {
          for (int j = 0; j < sample.cols; j++) {
            if (sample.at<uchar>(i, j) > 100) {
              count++;
            }
          }
        }
        q(i, j) = (count >= threshold) ? qr::white : qr::black;
      }
    }

    status.contributors_read.push_back(q);
    
    qr::computed_qr_t comp(q);

    auto list = comp.compute();

    status.contributors_img.push_back(transformed);
    
    status.dirtyflag = true;
  }

  return;
}
