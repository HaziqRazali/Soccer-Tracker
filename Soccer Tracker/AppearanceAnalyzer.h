#pragma once

#include <cv.h>
#include <vector>
#include "BallCandidate.h"
#include "globalSettings.h"

#include <iostream>

#include <opencv\cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

namespace st {

//*************************************************************************************************
// ----- This class is used for calculating the matching level between a ball candidate
// ----- and a part of an image
//*************************************************************************************************
class AppearanceAnalyzer {

	//_____________________________________________________________________________________________
	private:

		int eachTemplStripeH;

		Mat frame, restrictedArea;
		Mat teamA, teamB;
		Mat referee;

		vector<Mat> ballTempls;
		
		static const int matchMethod = CV_TM_CCORR_NORMED;
		// also such values as CV_TM_CCOEFF_NORMED, CV_TM_SQDIFF_NORMED can be used
		int _count = 0;

	//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		void setBallTempls (vector<Mat>& ballTempls) {
			this->ballTempls = ballTempls;
			if (!ballTempls.empty()) {
				eachTemplStripeH = fSize.y / ballTempls.size();
			}
		}
		//=========================================================================================
		void generateClassifier() {

			Mat temp;

			FileStorage aa("classifier.xml", FileStorage::READ);
			aa["some_name"] >> temp;
			aa.release();

			temp.row(0).copyTo(teamA);
			temp.row(1).copyTo(teamB);
			temp.release();

			FileStorage bb("Referee.xml", FileStorage::READ);
			bb["Referee"] >> temp;

			temp.copyTo(referee);

			temp.release();
		}

		//=========================================================================================
		void setFrame (Mat& frame) {
			this->frame = frame;
		}
		//=========================================================================================
		void setRestrictedArea (Mat& restrictedArea) {
			this->restrictedArea = restrictedArea;
		}

		//=========================================================================================
		void getMatches (BallCandidate* cand, int dotsCnt, vector<Point>& points, vector<double>& values, int TID = 0) {

			// Set limits
			Rect bounds(0, 0, frame.cols, frame.rows);

			points.clear();
			values.clear();

			// limit to valid space
			cand->curRect = cand->curRect & bounds;
	
			Mat crop = frame(cand->curRect);
			//Mat temp = crop.clone();
			Mat temp = crop;

			// Gay condition
			Mat RA_crop = restrictedArea(cand->curRect);

			// ---------- choose appropriate template ----------
			int templIdx = (cand->curCrd.y - 1) / eachTemplStripeH;
			
			Mat ballTempl = ballTempls[templIdx];

			cand->curTemplSize = ballTempl.rows;
			// ----------
			if (crop.rows < ballTempl.rows || crop.cols < ballTempl.cols) 
			{
				return;
			}

			// Player mask
			/*Mat restrictedMask = cand->restrictedMask;

			if (!restrictedMask.empty()) 
			{
				multiply(temp, restrictedMask, temp);
			}*/

			Mat cropGr, corrMtrx, cropCVar;
			matchTemplate(temp, ballTempl, corrMtrx, matchMethod);

			Mat cropResized    = temp(Rect(ballTempl.cols/2-1, ballTempl.rows/2-1, temp.cols-ballTempl.cols+1, temp.rows-ballTempl.rows+1));
			Mat RA_cropResized = RA_crop(Rect(ballTempl.cols/2-1, ballTempl.rows/2-1, temp.cols-ballTempl.cols+1, temp.rows-ballTempl.rows+1));

			multiply(corrMtrx, RA_cropResized, corrMtrx);

			double minVal, maxVal;
			Point minLoc, maxLoc;

			/*int count = 0;
			for (int i = 0; i < corrMtrx.rows; i++)
				for (int j = 0; j < corrMtrx.cols; j++)
				{
					if (corrMtrx.at<float>(i, j) > 0.94) count++;
				}

			if(TID == 0) cout << count << endl;*/

			// Finds the coordinate with the highest CC score
			for (int i = 0; i < dotsCnt; i++) 
			{
				minMaxLoc(corrMtrx, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
				corrMtrx.at<float>(maxLoc) = -2.0;
				maxLoc += Point(ballTempl.cols/2, ballTempl.rows/2) + cand->LU_Point;
				points.push_back(maxLoc);
				values.push_back(maxVal);
			}

			if (values.size() == 0 || points.size() == 0) {
				int x = 0;
			}
		}

		//=========================================================================================
		int getTeamID(Rect player)
		{
			Rect bounds(0, 0, frame.cols, frame.rows);
			Rect tempRect = player & bounds;

			// Resize Player to (20 x 45)
			Mat temp = frame(tempRect);
			Mat player_RGB = temp.clone();

			// Exit if Mat is empty
			if (player_RGB.empty()) return -1;

			// Else Classify
			else
			{
				// Resize Mat
				resize(player_RGB, player_RGB, Size(20, 45));

				// Extract RGB channels
				vector<Mat> temp_RGB_planes;
				split(player_RGB, temp_RGB_planes);

				// Convert all to row vector
				temp_RGB_planes[0] = temp_RGB_planes[0].reshape(0, 1).clone();
				temp_RGB_planes[1] = temp_RGB_planes[1].reshape(0, 1).clone();
				temp_RGB_planes[2] = temp_RGB_planes[2].reshape(0, 1).clone();

				// Compute feature vector
				Mat featureVector;
				hconcat(temp_RGB_planes, featureVector);

				// Convert featureVector to 32FC1
				featureVector.convertTo(featureVector, CV_32FC1);

				// Classify
				vector<float> dist;
				dist.push_back(norm(featureVector, teamA, NORM_L2));
				dist.push_back(norm(featureVector, teamB, NORM_L2));
				dist.push_back(norm(featureVector, referee, NORM_L2));

				return std::min_element(dist.begin(), dist.end()) - dist.begin();

				//// Classify
				//float distToA		= norm(featureVector, teamA, NORM_L2);
				//float distToB		= norm(featureVector, teamB, NORM_L2);
				//float distToReferee = norm(featureVector, referee, NORM_L2);

				//if (distToA < distToB) return 0;
				//else				   return 1;
			}
		}

		//=========================================================================================
		string type2str(int type) {
			string r;

			uchar depth = type & CV_MAT_DEPTH_MASK;
			uchar chans = 1 + (type >> CV_CN_SHIFT);

			switch (depth) {
			case CV_8U:  r = "8U"; break;
			case CV_8S:  r = "8S"; break;
			case CV_16U: r = "16U"; break;
			case CV_16S: r = "16S"; break;
			case CV_32S: r = "32S"; break;
			case CV_32F: r = "32F"; break;
			case CV_64F: r = "64F"; break;
			default:     r = "User"; break;
			}

			r += "C";
			r += (chans + '0');

			return r;
		}

		/*string ty = type2str(frame.type());
		printf("Matrix: %s %dx%d \n", ty.c_str(), frame.cols, frame.rows);*/

		//=========================================================================================
		vector<pair<float, Point>> evaluateContours(Point center)	{

			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			vector<pair<float, Point>> nC;

			nC.clear();

			// Set region of interest
			Point tl	= center - Point(30, 30);
			Point br	= center + Point(30, 30);
			Rect region = Rect(tl, br) & Rect(0, 0, frame.cols, frame.rows);
			Mat roi		= frame(region);

			// Image processing
			cvtColor(roi, roi, CV_BGR2GRAY);
			threshold(roi, roi, 128, 255, THRESH_BINARY);

			// Contour extraction
			findContours(roi, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

			// Contour Analysis.. Task -> set better area
			for (auto c : contours) if (contourArea(c) > 5)
			{
				// Compute circularity
				float circularity = pow(arcLength(c, true), 2) / (12.5664*contourArea(c));

				// Compute candidateCenter
				Rect br = boundingRect(Mat(c));
				Point candidateCenter(br.x + br.width / 2, br.y + br.height / 2);

				// Compute distance of candidateCenter to center of roi
				float dist = norm(Mat(center), Mat(candidateCenter), NORM_L2);

				// Update tempCandidate if circularity < 2
				if (circularity < 1.5) nC.push_back(make_pair(circularity, candidateCenter + region.tl()));
			}

			return nC;

		}

};

}