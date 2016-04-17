#pragma once

#include "globalSettings.h"
#include "KalmanFilter.h"
#include <vector>
#include <cv.h>

using namespace cv;
using namespace std;

namespace st {

//*************************************************************************************************
// ----- This class represents the entity of a player
//*************************************************************************************************
class PlayerCandidate {

	//_____________________________________________________________________________________________
	public:

		int startTrackTime, endTrackTime, lifeTime;
		int id;
		int updateTime;
		int predictTime;
		int teamID;

		bool Occlusion;
		bool Predict = false;

		float playerLikelihood;

		Mat image;

		vector<Point> coords;
		vector<Point> coordsKF;
		vector<Rect> prevRects;

		Point curCrd;
		Rect curRect;

		st::KalmanFilter KF;
		st::KalmanFilter smoothKF;

		bool ballAttached;

		//=========================================================================================
		PlayerCandidate (int time, Point crd, Rect rect, int teamID, bool Occlusion) {
			this->id = ID_counter++ * ID_groups_cnt + ID_shift;
			lifeTime = 0;

			// processNoiseCov, measureNoiseCov, errorCov
			KF = st::KalmanFilter(0.0001, 0.01, 0.01);
			smoothKF = st::KalmanFilter(0.00001, 0.01, 0.01);

			
			KF.initialize(crd);
			coordsKF.push_back(smoothKF.process(crd));

			this->startTrackTime = time;
			this->endTrackTime = -1;
			this->curCrd = crd;
			this->curRect = rect;
			this->teamID = teamID;
			this->Occlusion = Occlusion;
			
			updateTime = lifeTime;
			predictTime = 0;
			ballAttached = false;
		}

		//=========================================================================================
		bool compareDist (PlayerCandidate* pc1, PlayerCandidate* pc2) {
			double d1 = sqrt( ((curCrd.x - pc1->curCrd.x) * (curCrd.x - pc1->curCrd.x)) + ((curCrd.x-pc1->curCrd.y) * (curCrd.y - pc1->curCrd.y)) );
			double d2 = sqrt( ((curCrd.x - pc2->curCrd.x) * (curCrd.x - pc2->curCrd.x)) + ((curCrd.x-pc2->curCrd.y) * (curCrd.y - pc2->curCrd.y)) );
			return (d1 < d2);
		}

		//=========================================================================================
		void setRect (Rect nRect) {
			curRect = nRect;
			updateTime = lifeTime;
		}

		//=========================================================================================
		void setRect (Point nCrd) {
			if (!prevRects.empty()) {
				Rect r = prevRects[prevRects.size()-1];
				curRect = Rect(nCrd.x - r.width / 2, nCrd.y - r.height, r.width, r.height);
			}
			updateTime = lifeTime;
		}

		//=========================================================================================
		void setCrd (Point newCrd, bool predicted) {
			curCrd = newCrd;

			updateTime = lifeTime;

			if (predicted) 
			{
				predictTime++;
				Predict = true;
			} 
			
			else 
			{
				predictTime = 0;
				Predict = false;
			}
		}
		//=========================================================================================
		void updateStep () {
			coords.push_back(curCrd);
			coordsKF.push_back(smoothKF.process(curCrd));
			prevRects.push_back(curRect);
			lifeTime++;
			ballAttached = false;
		}

		//=========================================================================================
		~PlayerCandidate(void) {}
};

}