#pragma once

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <limits.h>

#include "xmlParser.h"
#include "globalSettings.h"

using namespace cv;
using namespace std;

namespace st {

//*************************************************************************************************
// ----- This class is used for fixing the provided ground truth data
//*************************************************************************************************
class TrajectoryAnalyzer {

	//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		TrajectoryAnalyzer(void) {}

		//=========================================================================================
		static void drawTrajectory (Mat& frame, vector<Point> ballPositions, int rad = 5, Scalar color = CV_RGB(0,0,0)) {

			frame = CV_RGB(255,255,255);

			if (ballPositions.empty())	return;

			Point prevPoint;
			bool startLine = false;
			for (unsigned i = 0; i < ballPositions.size(); i++) 
			{
				Point curPoint = ballPositions[i];

				if (curPoint != outTrajPoint) 
				{
					circle(frame, curPoint, rad, color);
					
					if (startLine) 
					{
						line(frame, prevPoint, curPoint, color);
					}

					startLine = true;
				} 

				else 
				{
					startLine = false;
				}

				prevPoint = curPoint;

			}
		}

		//=========================================================================================
		static void shiftTrajectory (vector<pair<int,Point>>& input, int shift) {
			
			for (auto& p : input) 
			{
				p.first += shift;
			}
		}

		//=========================================================================================
		static void scaleTrajectory (vector<pair<int, Point>>& traj, double scale) {
			for (auto& p : traj) {
				p.second = Point(int(scale * p.second.x), int(scale * p.second.y));
			}
		}

		//=========================================================================================
		static void readFullTrajectory (vector<vector<Point>>& trajectory, double scale, int startFrame, int endFrame) {
			
			trajectory.clear();

			for (int idx = 1; idx <= 6; idx++) 
			{
				vector<pair<int, Point>> xmlTraj, processedTraj;

				// Stores ball position in xmlTraj
				int res = xmlParser::parseBallPositions("ground_truth_ordered\\ground_truth_" + to_string(idx) + ".xgtf", xmlTraj);

				if (res)
				{
					cout << "parsing unsuccessfull" << endl;
					continue;
				}

				/*if (idx == 4 || idx == 5 || idx == 6) 
				{
					for (auto& bp : xmlTraj) 
					{
						bp.second = Point(1920-bp.second.x, bp.second.y);
					}
				}*/

				TrajectoryAnalyzer::scaleTrajectory(xmlTraj, scale);
				int shift = 0;

				switch (idx) 
				{
					case (1) : { shift = -5; break; } // -5
					case (2) : { shift = -5; break; } // -5
					case (3) : { shift = -5; break; } // -4
					case (4) : { shift = -5; break; } // -6
					case (5) : { shift = -5; break; } // -5
					case (6) : { shift = -5; break; } // -4
				}

				TrajectoryAnalyzer::shiftTrajectory(xmlTraj, shift);

				int length = endFrame - startFrame + 1;
				vector<Point> traj (length, outTrajPoint);
				
				// Error here
				for (auto& i : xmlTraj) 
				{
					//cout << idx << endl;
					int insertFrame = i.first;
					if (insertFrame <= endFrame && (insertFrame-startFrame > 0))
					{ 
						traj[insertFrame-startFrame] = i.second;
					}
					//cout << idx << endl;

				}

				trajectory.push_back(traj);

			}
		}

		//=========================================================================================
		~TrajectoryAnalyzer(void) {}

};

}