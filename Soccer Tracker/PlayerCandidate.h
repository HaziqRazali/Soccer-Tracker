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

		vector<Point> coords;
		vector<Rect> prevRects;

		Point curCrd;
		Rect curRect;

		st::KalmanFilter KF;

		bool ballAttached;

		//=========================================================================================
		PlayerCandidate (int time, Point crd, Rect rect) {
			this->id = ID_counter++ * ID_groups_cnt + ID_shift;
			lifeTime = 0;

			// processNoiseCov, measureNoiseCov, errorCov
			KF = st::KalmanFilter(0.0001, 0.01, 0.01);
			KF.initialize(crd);

			this->startTrackTime = time;
			this->endTrackTime = -1;
			this->curCrd = crd;
			this->curRect = rect;
			
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
				curRect = Rect(nCrd.x - r.width / 2, nCrd.y, r.width, r.height);
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
			} 
			
			else 
			{
				predictTime = 0;
			}
		}
		//=========================================================================================
		void updateStep () {
			coords.push_back(curCrd);
			//coordsKF.push_back(KF.process(curCrd));
			prevRects.push_back(curRect);
			lifeTime++;
			ballAttached = false;
		}

		//=========================================================================================
		~PlayerCandidate(void) {}
};

}