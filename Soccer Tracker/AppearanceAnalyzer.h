#pragma once

#include <cv.h>
#include <vector>
#include "BallCandidate.h"
#include "PlayerCandidate.h"
#include "globalSettings.h"
#include "BackGroundRemover.h"

#include <iostream>

#include <opencv\cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <stdlib.h>

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
		Mat tmplate[3];
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

			tmplate[0] = imread("White.png");
			tmplate[1] = imread("Blue.png");
			tmplate[2] = imread("Referee.png");
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

			}
		}

		void segmentPlayer(vector<PlayerCandidate*> pCandidates, Rect overlappedRoi, Mat& _frame, int TID, Mat _mask) {
			
			Rect bounds(0, 0, frame.cols, frame.rows);

			// Extract image of overlapping rect
			Mat _crop		= frame(overlappedRoi & bounds).clone();

			if (_crop.empty()) return;

			Mat mask = _mask(overlappedRoi & bounds);
			Mat crop;
			_crop.copyTo(crop, mask);

			Rect limit = Rect(0, 0, crop.cols, crop.rows);

			// Split into bgr
			vector<Mat> crop_bgr;
			split(crop, crop_bgr);

			// Template match
			for (int i = 0; i < pCandidates.size(); i++)
			{
				// Estimate rect of candidate
				Rect prevRect = Rect(0, 0, (10 * overlappedRoi.br().y / 540) + 30, (40 * overlappedRoi.br().y / 540) + 30) & limit;
				
				// Resize template to size of bounding box
				Mat playerTmplate = tmplate[pCandidates[i]->teamID].clone();
				resize(playerTmplate, playerTmplate, Size(prevRect.width, prevRect.height));	

				// Split template to bgr
				vector<Mat> playerTmplate_bgr;
				split(playerTmplate, playerTmplate_bgr);

				// Color based template matching
				Mat corrMtrx[3]; Point center;
				double score = 0;
				
				double minVal, maxVal;
				Point minLoc, maxLoc;
				for (int j = 0; j < 3; j++)
				{
					// Match template
					matchTemplate(crop_bgr[j], playerTmplate_bgr[j], corrMtrx[j], matchMethod);

					// Find best match
					minMaxLoc(corrMtrx[j], &minVal, &maxVal, &minLoc, &maxLoc);

					// Confirm theory
					center = center + maxLoc + Point(playerTmplate.cols/2, playerTmplate.rows/2);
					score = score + maxVal;
				}
				
				center = center / 3;
				pCandidates[i]->playerLikelihood = score / 3;
				pCandidates[i]->Occlusion = true;

				// Set image roi to zero for next iteration
				for (int j = 0; j < 3; j++)
				{
					Point tl = Point(center.x - playerTmplate.cols / 2, center.y - playerTmplate.rows / 2);
					Point br = Point(center.x + playerTmplate.cols / 2, center.y + playerTmplate.rows / 2);

					// Set region around best match to 0
					Mat roi = crop_bgr[j](Rect(tl, br));
					roi.setTo(0);
				}
				
				// top left and bottom right for candidate bb
				Point tl = Point(overlappedRoi.x + center.x - prevRect.width / 2, overlappedRoi.y + center.y - prevRect.height / 2);
				Point br = Point(overlappedRoi.x + center.x + prevRect.width / 2, overlappedRoi.y + center.y + prevRect.height / 2);

				Rect bb = Rect(tl, br);
				rectangle(_frame, Rect(tl, br), CV_RGB(255,0,0), 1);
				
				// candidate feet
				Point crd = Point(bb.x + bb.width/2, bb.y + bb.height);

				// Update pCandidate
				//*(pCandidates[i]->coords.end() - 1) = crd;
				pCandidates[i]->curCrd = crd;
				//*(pCandidates[i]->prevRects.end() - 1) = bb;
				pCandidates[i]->curRect = bb;
				pCandidates[i]->predictTime--;
			}
		}

		void setLimit(Rect input, Rect reference) {


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