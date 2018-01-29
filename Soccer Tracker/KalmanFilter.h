#pragma once

#include <opencv/highgui.h>
#include <opencv2/video/tracking.hpp>
#include <opencv/cv.h>
#include <vector>
#include "globalSettings.h"

using namespace cv;

namespace st {

//=================================================================================================
// ----- This class is used for proceeding Kalman filtration.
// ----- Is a wrapper for OpenCV version of Kalman Filter
//=================================================================================================
class KalmanFilter {

	//_____________________________________________________________________________________________
	private:
		
		cv::KalmanFilter KF;
		Mat_<float> state;
		Mat measure, estimated, prediction;

		bool initialized;

	//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		KalmanFilter (double processNoiseCov = 0.001, double measureNoiseCov = 0.05, double errorCov = 0.1, int measureCov = 1) {
			
			initialized = false;

			KF = cv::KalmanFilter(4, 2, 0);
			measure = Mat(2,1, CV_32F);
			measure.setTo(Scalar(1));

			KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   
														0,1,0,1,  
														0,0,1,0,  
														0,0,0,1);

			setIdentity(KF.measurementMatrix,   Scalar::all(measureCov));
			setIdentity(KF.processNoiseCov,     Scalar::all(processNoiseCov));
			setIdentity(KF.measurementNoiseCov, Scalar::all(measureNoiseCov));
			setIdentity(KF.errorCovPost,        Scalar::all(errorCov));
		}

		//=========================================================================================
	
		void initialize (Point2f p) {

			KF.statePre.at<float>(0) = p.x;
			KF.statePre.at<float>(1) = p.y;
			KF.statePre.at<float>(2) = 0.0;
			KF.statePre.at<float>(3) = 0.0;

			KF.statePost.at<float>(0) = p.x;
			KF.statePost.at<float>(1) = p.y;
			KF.statePost.at<float>(2) = 0.0;
			KF.statePost.at<float>(3) = 0.0;

			initialized = true;
		}

		//=========================================================================================
		Point2f process (Point2f p) {

			if (!initialized) {
				initialize(p);
			}

			prediction = KF.predict();

			measure.at<float>(0) = p.x;
			measure.at<float>(1) = p.y;
			estimated = KF.correct(measure);

			// ----- get the filtered position -----
			Point2f filteredP(estimated.at<float>(0), estimated.at<float>(1));
			return filteredP;
		}

		//=========================================================================================
		Point2f predict (bool debug = false) {
			prediction = KF.predict();
			return Point2f(prediction.at<float>(0), prediction.at<float>(1));
		}

		//=========================================================================================
		Point2f correct (Point2f p) {
			measure.at<float>(0) = p.x;
			measure.at<float>(1) = p.y;
			estimated = KF.correct(measure);
			return Point2f(estimated.at<float>(0), estimated.at<float>(1));
		}

		//=========================================================================================
		~KalmanFilter(void) {}

	};

}