#pragma once
#include <cv.h>
#include <tchar.h>
#include <string>
#include <ctime>

namespace st {

//*************************************************************************************************
// ----- This file stores global setting of the program, as well as enumerations and define flags
//*************************************************************************************************


cv::Point fSize, mSize;
cv::Point BALL_DRAW_RAD;
const cv::Point outTrajPoint(-1,-1);
int ID_counter, ID_shift, ID_groups_cnt;
int CAMERAS_CNT;
double scaleLoad = 0.5;
const int gui_camPreviewH = 1080, gui_camPreviewW = 1920;
const int gui_modelH = 652, gui_modelW = 948;

#define WRITE_VIDEO // save video to disk
#define DISPLAY_GROUND_TRUTH // display ground truth
#define NOT_FROM_THE_BEGINING // begin tracking from frame 361 (where the groundtruth starts)
#define START_FRAME 300 // 650 1300 2400
#define PLAYERS_KF // apply Kalman Filtering for players
#define WINDOW_PERSPECTIVE // take distance to the camera into the considiration

const int OUT_FRAME_RATE = 25; // frame rate for writing video
const int SLOW_MOTION_REPEAT_TIME = 20; // slows down the tracking

//*************************************************************************************************
enum BALL_STATE {
	INIT,
	SEARCHING,
	GOT_LOST,
	TRACKING,
	KICKED,
	ATTACHED_TO_PLAYER,
	SEPARATED_FROM_PLAYER,
	OUT_OF_FIELD,

	BALL_STATES_COUNT
};

//*************************************************************************************************
enum TRACKER_STATE {
	BALL_NOT_FOUND,
	BALL_FOUND,
	TRUE_POSITIVE_FOUND
};

//=================================================================================================
inline std::string getTimeString () {
	auto t = time(0);
	struct tm* now = localtime(&t);
	std::ostringstream s_stream;
	s_stream << (now->tm_year + 1900) << '.' << (now->tm_mon + 1) << '.'
		<< now->tm_mday << '-' << now->tm_hour << '-' << now->tm_min << '-' << now->tm_sec;
	return s_stream.str();
}

//=================================================================================================
inline void createFolder (std::string folderName) {
	std::wstring b = std::wstring(folderName.begin(), folderName.end());
	const wchar_t* a = b.c_str();
	_tmkdir(a);
}

}