CMAKE_MINIMUM_REQUIRED( VERSION 2.8.3 )
PROJECT( object_detect ) #设定工程名

ADD_DEFINITIONS(-DLINUX_USE)
set(CMAKE_BUILD_TYPE Release)
#set(OpenCV_DIR /usr/local/opencv32/share/OpenCV/)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

#设定存放编译出来的库文件的目录
SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../)

#message(${OpenCV_INCLUDE_DIRS})

#设定头文件目录
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include 
	#			${OpenCV_INCLUDE_DIRS}
			${CUDA_INCLUDE_DIRS})

AUX_SOURCE_DIRECTORY(./src DARK_DIR)
ADD_LIBRARY(darknet SHARED ${DARK_DIR})
target_link_libraries(darknet m  pthread)# ${OpenCV_LIBS})
