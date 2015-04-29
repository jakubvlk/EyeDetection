add_executable( EyeDetection main.cpp findIrisCentreWithCorrectionOfEyeROI.cpp)
target_link_libraries(EyeDetection ${OpenCV_LIBS} )