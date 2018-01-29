#pragma once

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

namespace st {

//*************************************************************************************************
// ----- A wrapper class used for reading video from the file
//*************************************************************************************************
class VideoReader {

	//_____________________________________________________________________________________________
	private:
	
		vector<string> videoNames;
		vector<VideoCapture> videoCaptures;

	//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		VideoReader(void) {}

		//=========================================================================================
		VideoCapture addVideo (string fileName) {

			VideoCapture vidCap(fileName);
			videoCaptures.push_back(vidCap);
			//videoCaptures.push_back(fileName);

			return vidCap;
		}

		//=========================================================================================
		int getCodecExt () {
			int ex = -1;
			if (!videoCaptures.empty()) {
				ex = static_cast<int>(videoCaptures[0].get(CV_CAP_PROP_FOURCC));
			}
			return ex;
		}

		//=========================================================================================
		vector<VideoCapture> getCaptures () {
			return videoCaptures;
		}

		//=========================================================================================
		~VideoReader(void) {}
};

}