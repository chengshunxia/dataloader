OPTS=-g
BOOST_OPTS=-lboost_filesystem -lboost_system -lboost_chrono -lboost_thread -lboost_numpy3 -lboost_date_time -lboost_python3
OPENCV_OPTS=-lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_datasets -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_line_descriptor -lopencv_optflow -lopencv_video -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_rgbd -lopencv_viz -lopencv_surface_matching -lopencv_text -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core
PYTHON_INCLUDE_PATH=-I /usr/include/python3.6m/
PYTHON3_OPTS=-lpython3.6m
OPENCV_OPTS=$(pkg-config --libs opencv)


.PHONY:all

all:
	g++ ${OPTS} -o datasets.o -c datasets.cpp  ${BOOST_OPTS} ${OPENCV_OPTS} ${PYTHON_INCLUDE_PATH}
	g++ ${OPTS} -o transforms.o -c transforms.cpp  ${BOOST_OPTS} ${OPENCV_OPTS} ${PYTHON_INCLUDE_PATH}
	g++ ${OPTS} -o dataloader.o -c dataloader.cpp  ${BOOST_OPTS} ${OPENCV_OPTS} ${PYTHON_INCLUDE_PATH}
	g++ ${OPTS} -o test test.cpp datasets.o transforms.o dataloader.o ${PYTHON_INCLUDE_PATH} ${BOOST_OPTS} ${PYTHON3_OPTS} ${OPENCV_OPTS} -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_datasets -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_line_descriptor -lopencv_optflow -lopencv_video -lopencv_plot -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_rgbd -lopencv_viz -lopencv_surface_matching -lopencv_text -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core