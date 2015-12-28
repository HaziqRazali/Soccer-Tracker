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
		void setFrame (Mat& frame) {
			this->frame = frame;
		}
		//=========================================================================================
		void setRestrictedArea (Mat& restrictedArea) {
			this->restrictedArea = restrictedArea;
		}

		//=========================================================================================
		void getMatches (BallCandidate* cand, int dotsCnt, vector<Point>& points, vector<double>& values) {

			// Set limits
			Rect bounds(0, 0, frame.cols, frame.rows);

			points.clear();
			values.clear();

			// limit to valid space
			cand->curRect = cand->curRect & bounds;
	
			Mat crop = frame(cand->curRect);

			// Dunno what this is for
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

			Mat restrictedMask = cand->restrictedMask;
			if (!restrictedMask.empty()) 
			{
				multiply(crop, restrictedMask, crop);
			}

			Mat cropGr, corrMtrx, cropCVar;
			matchTemplate(crop, ballTempl, corrMtrx, matchMethod);

			Mat cropResized    = crop(Rect(ballTempl.cols/2-1, ballTempl.rows/2-1, crop.cols-ballTempl.cols+1, crop.rows-ballTempl.rows+1));
			Mat RA_cropResized = RA_crop(Rect(ballTempl.cols/2-1, ballTempl.rows/2-1, crop.cols-ballTempl.cols+1, crop.rows-ballTempl.rows+1));

			multiply(corrMtrx, RA_cropResized, corrMtrx);

			double minVal, maxVal;
			Point minLoc, maxLoc;

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