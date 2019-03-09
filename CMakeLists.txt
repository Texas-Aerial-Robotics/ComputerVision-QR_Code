cmake_minimum_required(VERSION 3.10)
project( qrpart )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

find_package( OpenCV 3.3 REQUIRED )

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable( qrpart main_ros.cpp src/qr_br.cpp src/cvlocalizer.cpp )
target_link_libraries( qrpart ${catkin_LIBRARIES} ${OpenCV_LIBS} )

install(TARGETS qrpart
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
