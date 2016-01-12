#pragma once

#include <utility>
#include <string>
#include <deque>

namespace st {

//*************************************************************************************************
// ----- This structure is used for calculating TN, TP, FP, FN, R and distance metrics
// ----- for comparing tracking results with the ground truth values
//*************************************************************************************************
struct AccuracyMetric {

	int TP, TN, FP, FN;
	int Total;
	vector<pair<Point, Point>> results;
	double TR;
	double TP_norm, TN_norm, FP_norm, FN_norm, R, dist_norm;
	deque<double> _dist;

	//=============================================================================================
	AccuracyMetric () {
		TP = 0; TN = 0; FP = 0; FN = 0; dist_norm = 0;
	}

	//=============================================================================================
	void addDistance (double dist) {

		_dist.push_back(dist);

		if (_dist.size() > 15)	_dist.pop_front();
	}

	//=============================================================================================
	void normalize () {

		double sum = double(TP + TN + FP + FN);

		if (sum == 0) 
		{
			TP_norm = 0.0; TN_norm = 0.0; FP_norm = 0.0; FN_norm = 0.0;
		}

		if (TP + TN == 0) 
		{
			R = 0.0;
		} 
		else 
		{
			R = double(TP) / double(TP + TN) * 100;
		}

		TP_norm = double(TP) / sum * 100;
		TN_norm = double(TN) / sum * 100;
		FP_norm = double(FP) / sum * 100;
		FN_norm = double(FN) / sum * 100;

		if (_dist.size() > 0) 
		{
			dist_norm = 0;

			for (auto& l : _dist) 
			{
				dist_norm += l;
			}

			dist_norm /= _dist.size();
		} 

		else 
		{
			dist_norm = 0;
		}
	}

	//=============================================================================================
	std::string toString_prc () {
		normalize();
		char buffer[100];
		//sprintf(buffer, "TP: %2.1f, TN: %2.1f, FP: %2.1f, FN: %2.1f, R: %2.1f, D: %.1f", TP_norm, TN_norm, FP_norm, FN_norm, R, dist_norm);
		sprintf(buffer, "TP: %d / %d", TP, Total);
		return buffer;
	}
};

}