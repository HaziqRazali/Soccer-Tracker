#pragma once

#include <cv.h>
#include <vector>
#include <iterator>

using namespace cv;
using namespace std;

namespace st {

//********** the struct for describing one peak
struct Peak {
	int crd_L, crd_R, crd_C;
	float val_L, val_R, val_C;
	float area_L, area_R, area_S;
	float sharpness_L, sharpness_R, sharpness;
	bool operator< (const Peak& a) { return this->val_C < a.val_C; }
};

//*************************************************************************************************
// ----- This class is used for calculation histogram and processing it (finding peaks, intervals, etc.)
//*************************************************************************************************
class Histogrammer {

	//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		Histogrammer () {}

		//=========================================================================================
		//static MatND getHist (Mat& image, int channel, int binsCnt, bool norm = false, bool filter = true) {
		//	MatND hist;
		//	int channels[1] = {channel}; 
		//	int numOfBins[1] = {binsCnt};
		//	float hrangers[2] = {0.0, 255.0};
		//	const float* ranges[1] = {hrangers};
		//	calcHist(&image, 1, channels, Mat(), hist, 1, numOfBins, ranges);
		//	if (norm) {
		//		normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
		//	}
		//	if (filter) {
		//		hist = filterHist(hist);
		//	}
		//	return hist;
		//}

		//=========================================================================================
		static MatND filterHist (MatND& hist) {
		
			MatND histF;

			int bins = hist.rows;
			int radius = int(bins / 100) * 2 + 1;
			double sigma = double(radius)/3;

			GaussianBlur(hist, histF, Size(1, radius), sigma);
			return histF;
		}

		//=========================================================================================
		static MatND getHistMask (MatND& hist) {
			int bins = hist.rows;
			vector<int> extramas(bins,0);
			for (int i = 1; i < bins-1; i++) {
				float l = hist.at<float>(i-1);
				float c = hist.at<float>(i);
				float r = hist.at<float>(i+1);
				if ((l < c) && (c > r)) { // local max
					extramas[i] = 1;
				} else if ((l >= c) && (c <= r)) { // local min
					extramas[i] = -1;
				}
			}
			// ========== find all peaks in the histogram
			vector<Peak> peaks;
			for (int i = 0; i < bins; i++) {
				if (extramas[i] == 1) { // it's a peak;
					// ----- create new peak
					Peak peak;
					peak.crd_C = i;
					peak.val_C = hist.at<float>(i);
					// ----- find left and right side of the peak
					int left = i, right = i;
					while ((left > 0) && (extramas[left] != -1)) left--;
					while ((right < bins-1) && (extramas[right] != -1)) right++;
					peak.crd_L = left;
					peak.crd_R = right;
					peak.val_L = hist.at<float>(left);;
					peak.val_R = hist.at<float>(right);
					// ----- compute sharpness of the peak
					peak.sharpness_L = (peak.val_C - peak.val_L) / float(peak.crd_C - peak.crd_L);
					peak.sharpness_R = (peak.val_C - peak.val_R) / float(peak.crd_R - peak.crd_C);
					peak.sharpness = std::min(peak.sharpness_L, peak.sharpness_R);
					// ----- compute area of the peak
					float area = 0.0;
					for (int j = left; j <= i; j++) {
						area += hist.at<float>(j);
					}
					peak.area_L = area;
					area = 0.0;
					for (int j = i; j <= right; j++) {
						area += hist.at<float>(j);
					}
					peak.area_R = area;
					peak.area_S = peak.area_L + peak.area_R - peak.val_C;
					// ----- push the new peak into the vector of all peaks
					peaks.push_back(peak);
				}
			}
			// ========== select peaks that would be removed
			vector<Peak> peaksToRemove;
			if (peaks.size() > 0) {
				// ----- select the peak with maximum value;
				vector<Peak>::iterator it = std::max_element(peaks.begin(), peaks.end());
				Peak maxPeak = *it;
				peaksToRemove.push_back(maxPeak);
				// ----- select all other proper peaks
				for (unsigned i = 0; i < peaks.size(); i++) {
					Peak peak = peaks[i];
					int peakDist = abs(peak.crd_C - maxPeak.crd_C);
					//if ((peakDist < 0.1*bins) && (peak.val_C > 0.5 * maxPeak.val_C) && (peak.sharpness >= 0.3 * maxPeak.sharpness)) {
					if ((peakDist < 0.1*bins) && (peak.val_C > 0.5 * maxPeak.val_C)) {
						peaksToRemove.push_back(peak);
					}
				}
			}
			//modifyPeaks(hist, peaksToRemove, 0.05f);
			// ========== prepare intervals that would be removed form the histogram
			MatND histMask = MatND::ones(bins, 1, CV_32FC1);
			for (unsigned i = 0; i < peaksToRemove.size(); i++) {
				int leftSide = peaksToRemove[i].crd_L;
				int rightSide = peaksToRemove[i].crd_R;
				for (int j = leftSide; j <= rightSide; j++) {
					histMask.at<float>(j) = 0.0;
				}
			}
			// ========= return the mask
			return histMask;
		}

		//=========================================================================================
		static void getHistMask2 (MatND& hist, vector<int>& intervalsL, vector<int>& intervalsR) {
			int bins = hist.rows;
			vector<int> extramas(bins,0);
			for (int i = 1; i < bins-1; i++) {
				float l = hist.at<float>(i-1);
				float c = hist.at<float>(i);
				float r = hist.at<float>(i+1);
				if ((l < c) && (c > r)) { // local max
					extramas[i] = 1;
				} else if ((l >= c) && (c <= r)) { // local min
					extramas[i] = -1;
				}
			}
			// ========== find all peaks in the histogram
			vector<Peak> peaks;
			for (int i = 0; i < bins; i++) {
				if (extramas[i] == 1) { // it's a peak;
					// ----- create new peak
					Peak peak;
					peak.crd_C = i;
					peak.val_C = hist.at<float>(i);
					// ----- find left and right side of the peak
					int left = i, right = i;
					while ((left > 0) && (extramas[left] != -1)) left--;
					while ((right < bins-1) && (extramas[right] != -1)) right++;
					peak.crd_L = left;
					peak.crd_R = right;
					peak.val_L = hist.at<float>(left);;
					peak.val_R = hist.at<float>(right);
					// ----- compute sharpness of the peak
					peak.sharpness_L = (peak.val_C - peak.val_L) / float(peak.crd_C - peak.crd_L);
					peak.sharpness_R = (peak.val_C - peak.val_R) / float(peak.crd_R - peak.crd_C);
					peak.sharpness = std::min(peak.sharpness_L, peak.sharpness_R);
					// ----- compute area of the peak
					float area = 0.0;
					for (int j = left; j <= i; j++) {
						area += hist.at<float>(j);
					}
					peak.area_L = area;
					area = 0.0;
					for (int j = i; j <= right; j++) {
						area += hist.at<float>(j);
					}
					peak.area_R = area;
					peak.area_S = peak.area_L + peak.area_R - peak.val_C;
					// ----- push the new peak into the vector of all peaks
					peaks.push_back(peak);
				}
			}
			// ========== select peaks that would be removed
			vector<Peak> peaksToRemove;
			if (peaks.size() > 0) {
				// ----- select the peak with maximum value;
				vector<Peak>::iterator it = std::max_element(peaks.begin(), peaks.end());
				Peak maxPeak = *it;
				//peaksToRemove.push_back(maxPeak);
				// ----- select all other proper peaks
				for (unsigned i = 0; i < peaks.size(); i++) {
					Peak peak = peaks[i];
					int peakDist = abs(peak.crd_C - maxPeak.crd_C);
					//if ((peakDist < 0.1*bins) && (peak.val_C > 0.5 * maxPeak.val_C) && (peak.sharpness >= 0.3 * maxPeak.sharpness)) {
					if ((peakDist < 0.1*bins) && (peak.val_C > 0.5 * maxPeak.val_C)) {
						peaksToRemove.push_back(peak);
					}
				}
			}
			//modifyPeaks(hist, peaksToRemove, 0.01f);

			// ========== prepare intervals that would be removed form the histogram
			vector<int> intervals;
			for (unsigned i = 0; i < peaksToRemove.size(); i++) {
				intervals.push_back(peaksToRemove[i].crd_L);
				intervals.push_back(peaksToRemove[i].crd_R);
			}
			mergeIntervals(intervals, intervalsL, intervalsR);
		}

		//=========================================================================================
		static inline void mergeIntervals (vector<int>& intervals, vector<int>& intervalsL, vector<int>& intervalsR) {
			// of course intervalsL and intervalsR should be the same size
			for (auto it = intervals.begin(); it < intervals.end()-1;) {
				if (*it == *(it+1)) {
					it = intervals.erase(it);
					it = intervals.erase(it);
				} else {
					++it;
				}
			}
			intervalsL.clear();
			intervalsR.clear();
			bool odd = true;
			for (auto& i : intervals) {
				if (odd) {
					intervalsL.push_back(i);
				} else {
					intervalsR.push_back(i);
				}
				odd = !odd;
			}
		}

		//=========================================================================================
		static void modifyPeaks (MatND& hist, vector<Peak>& peaks, float coeff) {
			// ========== modify peaks if needed (make them wider or thinner)
			int bins = hist.rows;
			for (unsigned i = 0; i < peaks.size(); i++) {
				// get the left and right coords
				int left = peaks[i].crd_L;
				int right = peaks[i].crd_R;
				float area = float(peaks[i].area_S * abs(coeff));
				bool direction = coeff > 0;
				if (direction) {
					// ---------- make peak wider
					while (area > 0) {
						if ((left == 0) && (right == bins-1)) {
							break;
						}
						float leftVal = hist.at<float>(left);
						float rightVal = hist.at<float>(right);
						if ((left == 0) || ((right < bins-1) && (rightVal > leftVal))) {
							right++;
							area -= rightVal;
						} else {
							left--;
							area -= leftVal;
						}
					}
				} else {
					// ---------- make peak thinner
					int center = peaks[i].crd_C;
					while (area > 0) {
						if ((left == center) && (right == center)) {
							break;
						}
						float leftVal = hist.at<float>(left);
						float rightVal = hist.at<float>(right);
						if ((left == center) || ((right > center) && (rightVal < leftVal))) {
							right--;
							area -= rightVal;
						} else {
							left++;
							area -= leftVal;
						}
					}
				}
				// wright back the new values;
				peaks[i].crd_L = left;
				peaks[i].crd_R = right;
			}
		}

		//=========================================================================================
		static Mat getHistImage (MatND& hist, MatND& mask) {
			int bins = hist.rows;
			double minVal = 0, maxVal = 0;

			minMaxLoc(hist, &minVal, &maxVal);
			Mat histImg (bins, bins, CV_8UC3, CV_RGB(255, 255, 255));

			int hpt = int(0.9*bins);
			for (int h = 0; h < bins; h++) {
				float binVal = hist.at<float>(h);
				int intens = int(binVal*hpt/maxVal);
				int cc = int(double(1.0-mask.at<float>(h)) * 255 * 0.5);
				line(histImg, Point(h, bins), Point(h, bins-intens), CV_RGB(cc, cc, cc));
			}
			return histImg;
		}

		//=========================================================================================
		static Mat backProj (Mat& src, MatND& hist, int channel, bool treshold = false) {
			const float* ranges[1];
			float hrangers[2] = {0.0, 255.0};
			ranges[0] = hrangers;
			int channels[1] = {channel};
			MatND histN;

			normalize(hist, histN, 0, 255, NORM_MINMAX, -1, Mat());

			MatND backproj;
			calcBackProject(&src, 1, channels, histN, backproj, ranges);

			if (treshold) {
				threshold(backproj, backproj, 0, 255, THRESH_BINARY);
			}

			return backproj;
		}

		//=========================================================================================
		~Histogrammer (void) {}

};

}