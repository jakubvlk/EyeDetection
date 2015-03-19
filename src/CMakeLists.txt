add_executable( EyeDetection main.cpp captureImage.cpp captureImage.h detection.cpp detection.h constants.h)
target_link_libraries(EyeDetection ${OpenCV_LIBS} )