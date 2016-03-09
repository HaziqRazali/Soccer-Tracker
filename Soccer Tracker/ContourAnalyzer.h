#pragma once

#include "globalSettings.h"

#include <cv.h>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define PI 3.1415926535

using namespace cv;
using namespace std;

namespace st {

//*************************************************************************************************
// ----- This class is used to perform contour analysing on the binary mask retrieved
// ----- from B/F segmentation step
//*************************************************************************************************
class ContourAnalyzer {

	//_____________________________________________________________________________________________
	private:

		double expectedBallSize[2], expectedPlayerSize[2];

	//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		ContourAnalyzer () {
			expectedPlayerSize[0] = 0.04;
			expectedPlayerSize[1] = 0.1;
			expectedBallSize[0] = 0.003;
			expectedBallSize[1] = 0.014; // Default -> 0.01. For offside handling -> 0.014
		}

		//=========================================================================================
		void process (const Mat& binMask, vector<Rect>& players, vector<Point>& ball) {
			vector<vector<Point>> contours;
			findContours(binMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			//findContours(image, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

			vector<vector<Point>> players_cand, ball_cand;
			filterAndSortRoi_Geom(contours, players_cand, ball_cand);
			
			//drawContours(frame, players_cand, -1, CV_RGB(255, 0, 0), 4);
			//drawContours(frame, ball_cand, -1, CV_RGB(0, 0, 255), 4);
			
			vector<vector<Point>>::const_iterator it = players_cand.begin();
			while (it != players_cand.end()) {
				players.push_back(boundingRect(*it));
				++it;
			}
			it = ball_cand.begin();
			while (it != ball_cand.end()) {
				Rect r = boundingRect(*it);
				Point p (r.x + r.width/2, r.y + r.height/2);
				ball.push_back(p);
				++it;
			}
		}

		//=========================================================================================
		void filterAndSortRoi_Geom (vector<vector<Point>>& roi, vector<vector<Point>>& player, vector<vector<Point>>& ball) {
			
			int pxExpectedPlayerSize[2] = {int(expectedPlayerSize[0] * fSize.y), int(expectedPlayerSize[1] * fSize.x)};
			int pxExpectedBallSize[2] = {int(expectedBallSize[0] * fSize.y), int(expectedBallSize[1] * fSize.x)};
			
			vector<vector<Point>>::const_iterator it = roi.begin();
		
			while (it != roi.end()) 
			{

				Rect boundRect = boundingRect(*it);
				
				// Compute area
				double area = contourArea(*it);

				// Compute perimeter
				int perimeter = int(it->size());
				double perimeter_ = arcLength(*it, true); 

				// Compute roundness
				Point2f circleCntr;
				float circleRad;
				minEnclosingCircle(*it, circleCntr, circleRad);
				double roundness = 4 * PI * area / (perimeter * perimeter);

				// Condition for player
				if ((boundRect.height > pxExpectedPlayerSize[0]) && (boundRect.height < pxExpectedPlayerSize[1]) &&
					(boundRect.height > boundRect.width) && (area > 0.3 * double(boundRect.area()))		) 
				{
					player.push_back(*it);
				} 
				
				// Condition for ball
				else if ((boundRect.height > pxExpectedBallSize[0]) && (boundRect.height < pxExpectedBallSize[1]) &&
						 (boundRect.height < 2 * boundRect.width) && (boundRect.width < 3 * boundRect.height) &&
						 (area > 0.2 * double(boundRect.area())) && (roundness > 0.4)) 
				{
					ball.push_back(*it);
				}

				++it;
			}
		}

		//=========================================================================================
		//void filterAndSortRoi_Appearance (const Mat& frameFull, const Mat& maskFull, vector<vector<Point>>& ball_cand) {
		//	vector<vector<Point>>::const_iterator it = ball_cand.begin();
		//	while (it != ball_cand.end()) {
		//		Rect bRectScaled = boundingRect(*it);
		//		Rect bRectFull = Rect(invScale*bRectScaled.x, invScale*bRectScaled.y, invScale*bRectScaled.width, invScale*bRectScaled.height);
		//		Mat frameFull_ROI = frameFull(bRectFull);
		//		Mat maskFull_ROI = maskFull(bRectFull);
		//		int nonZero = countNonZero(maskFull_ROI);
		//		Mat maskedROI, maskedROI_grey;
		//		frameFull_ROI.copyTo(maskedROI, maskFull_ROI);
		//		cvtColor(maskedROI, maskedROI_grey, CV_BGR2GRAY);
		//		maskedROI_grey.convertTo(maskedROI_grey, CV_32F);
		//		vector<Mat> maskedROI_channels = vector<Mat>(3, Mat());
		//		split(maskedROI, maskedROI_channels);
		//		maskedROI_grey = maskedROI_grey / 255;
		//		vector<double> colorVarience = vector<double>(3, 0.0);
		//		for (int i = 0; i < 3; i++) {
		//			Mat channel;
		//			maskedROI_channels[i].convertTo(channel, CV_32F);
		//			pow(channel/255 - maskedROI_grey, 2, channel);
		//			colorVarience[i] = sum(channel)[0] / nonZero;
		//		}
		//		double maxColorVarience = max(colorVarience[0], max(colorVarience[1], colorVarience[2]));
		//
		//		Scalar s = sum(maskedROI);
		//		//double colorIntens = (s[0] + s[1] + s[2]) / (contourArea(*it) * 3 * 255);
		//		double colorIntens = (s[0] + s[1] + s[2]) / (nonZero * 3 * 255);
		//		//				double colorIntens2 = (s[0] + s[1] + s[2]) / countNonZero(smallMask);
		//
		//		//if (colorIntens < 0.8) {
		//		if (maxColorVarience > 0.01) {
		//			it = ball_cand.erase(it);
		//		} else {
		//			++it;
		//		}
		//		//imshow("t", maskedROI_grey);
		//	}
		//}

		//=========================================================================================
		~ContourAnalyzer(void) {}
};

}