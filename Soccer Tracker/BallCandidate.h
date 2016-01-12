#pragma once

#include "globalSettings.h"
#include "KalmanFilter.h"
#include <cv.h>
#include <vector>
#include<iostream>

using namespace cv;
using namespace std;

namespace st {

//*************************************************************************************************
// ----- This class represents the structure of ball candidate
//*************************************************************************************************
class BallCandidate {

	//_____________________________________________________________________________________________
	private:

		int startTrackTime, endTrackTime;
		BALL_STATE curState;
		vector<pair<int, BALL_STATE>> stateSwitches;
		bool curStateSaved;

	//_____________________________________________________________________________________________
	public:

		int lifeTime, id, curTemplSize;
		Point curCrd, curRad, LU_Point, RD_Point;
		Point predCrd;
		Rect curRect;
		double curAppearM;
		double curCirc;
		vector<Point> coords, coordsKF;
		vector<Point> winRads;
		vector<double> appearM;
		vector<BALL_STATE> states;

		int updateTime;
		int predictTime;

		Mat restrictedMask;

		int looseTime, findTime;
		int attachedToPlayer;
		double attachedHeight;

		st::KalmanFilter KF;

		//=========================================================================================
		BallCandidate (int time, Point crd, Point winRad, double prob) {
			this->id = ID_counter++ * ID_groups_cnt + ID_shift;

			// processNoiseCov, measureNoiseCov, errorCov, 
			KF = st::KalmanFilter(0.0001, 0.01, 0.01); // What I used for Brute Force Prediction -> KF = st::KalmanFilter(0.1, 0.01, 0.01, 1);
			KF.initialize(crd);

			this->startTrackTime = time;
			this->endTrackTime = -1;
			this->curCrd = crd;
			this->curRad = winRad;
			this->curAppearM = prob;

			fitFrame();

			coords.push_back(crd);
			coordsKF.push_back(KF.process(crd));
			winRads.push_back(winRad);
			appearM.push_back(prob);

			predictTime = 0;

			lifeTime = 0;
			looseTime = 0;
			findTime = 0;
			attachedToPlayer = -1;

			curState = BALL_STATE::INIT;

			restrictedMask = Mat();

			curStateSaved = false;

		}

		//=========================================================================================
		static bool compareLastAppearM (BallCandidate* bc1, BallCandidate* bc2) {
			return (bc1->curAppearM > bc2->curAppearM);
		}

		//=========================================================================================
		static bool compareAppearMSum (BallCandidate* bc1, BallCandidate* bc2) {
			int lh = 5;
			return (bc1->getAppearMSum(lh) > bc2->getAppearMSum(lh));
		}

		//=========================================================================================
		double getAppearMSum (int lh = -1) {
			// ---------- get the sum of appearance correlation for the last <lh> frames ----------
			int start = 0;
			if (lh != -1) {
				start = coords.size()-lh;
				start = int(max(0.0, double(start)));
			}
			double dd = 0;
			for (int i = start+1; i < int(coords.size()); i++) {
				dd += appearM[i];
			}
			if (appearM.size() > 1) {
				dd /= (appearM.size()-start-1);
			}
			return dd;
		}

		//=========================================================================================
		BALL_STATE getState () {
			return curState;
		}

		//=========================================================================================
		void switchState (int time, BALL_STATE nState) {
			// ---------- switch ball candidate from current state to a new one ----------
			states.push_back(curState);
			curStateSaved = true;
			curState = nState;
			stateSwitches.push_back(pair<int, BALL_STATE> (time, nState));
		}

		//=========================================================================================
		int getLastStateDuration (int curTime) {

			if (stateSwitches.empty()) 
			{
				return -1;
			}

			return curTime - stateSwitches[stateSwitches.size()-1].first;
		}

		//=========================================================================================
		int getStateDuration (BALL_STATE _state)
		{
			int counter = 0;

			// Loop though all states
			for (auto& st : states) 
			{
				if (st == _state) 
				{
					counter++;
				}
			}
			return counter;
		}

		//=========================================================================================
		int getStateDuration (vector<BALL_STATE> _states) {
			
			int counter = 0;
			for (auto& st : states) 
			{
				for (auto& _st : _states) 
				{
					if (st == _st) {
						counter++;
					}
				}
			}
			return counter;
		}

		//=========================================================================================
		void updateStep (int boost = 1) {
			// ---------- save all made changes ----------
			if (!curStateSaved) 
			{
				states.push_back(curState);
			}

			if (boost > 1) for (int i = 0; i < boost; i++) states.push_back(curState);

			fitFrame();

			// Update coordinates
			coords.push_back(curCrd);
			coordsKF.push_back(KF.process(curCrd));

			winRads.push_back(curRad);
			appearM.push_back(curAppearM);

			// Update lifeTime
			lifeTime++;
			restrictedMask = Mat();
			curStateSaved = false;
		}

		//=========================================================================================
		void fitFrame () {
			// ---------- calculate the precise rectangle for search zone ----------
			int left   = curCrd.x - curRad.x;
			int right  = curCrd.x + curRad.x;
			int up     = curCrd.y - curRad.y;
			int down   = curCrd.y + curRad.y;

			up = int(std::max(double(up), 0.0));
			down = int(std::min(double(down), double(fSize.y-1)));
			left = int(std::max(double(left), 0.0));
			right = int(std::min(double(right), double(fSize.x-1)));

			LU_Point = Point(left, up);
			RD_Point = Point(right, down);

			curRect =  Rect(LU_Point, RD_Point);
		}

		//=========================================================================================
		void finishTracking (int time) {
			endTrackTime = time;
		}
};


}