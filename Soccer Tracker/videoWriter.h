#pragma once

#include <cv.h>
#include <opencv2\opencv.hpp>
#include <highgui.h>
#include <vector>

using namespace std;

namespace st {

//*************************************************************************************************
// ----- A wrapper class used for writing video to the file
//*************************************************************************************************
class VideoWriter {

	//_____________________________________________________________________________________________
	private:

		std::vector<cv::VideoWriter> outputVideo;
		std::vector<int> videoID;
		std::string folderName;

	//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		VideoWriter (string folderName) {
			this->folderName = folderName;
		}
	
		//=========================================================================================
		cv::VideoWriter addVideo (std::string fileName, cv::Size s, int frameRate, int ex, int _videoID) {
			cv::VideoWriter vw(folderName + fileName, ex, frameRate, s, true);
			if (!vw.isOpened())	{
				cout  << "Could not open the output video for write" << endl;
			}
			outputVideo.push_back(vw);
			videoID.push_back(_videoID);
			return vw;
		}

		//=========================================================================================
		cv::VideoWriter getVideoWriter (int _videoID) {
			int idx = std::distance(videoID.begin(), std::find(videoID.begin(), videoID.end(), _videoID));
			return outputVideo[idx];
		}

		//=========================================================================================
		~VideoWriter (void) {}
};

}