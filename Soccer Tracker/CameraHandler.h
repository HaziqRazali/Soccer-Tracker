#pragma once

#include <opencv/highgui.h>
#include <vector>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include "globalSettings.h"
#include "Configurator.h"
#include "VideoReader.h"
#include "TemplateGenerator.h"

using namespace cv;
using namespace std;

namespace st {

//*************************************************************************************************
// ----- Represents entity of one camera 
//*************************************************************************************************
struct Camera {

	Point3d camCoords;
	int id, idx;
	bool viewHFlip, viewVFlip, projHFlip, projVFlip;
	//Point crd, videoResolution;
	Mat homography;
	VideoCapture capture;
	Rect viewRect;
	Scalar backGrColor;
	vector<Mat> ballTemplates;
	int hueIntervalL, hueIntervalR;
	double perspectiveRatio;
	Camera () {}
};

//*************************************************************************************************
// ---- This class stores all information about cameras, receives it by reading configuration file
//*************************************************************************************************
class CameraHandler {
	
	//_____________________________________________________________________________________________
	private:

		Configurator* configurator;
		VideoReader* vidReader;
		vector<Camera*> cameras;

	//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		CameraHandler (Configurator* c, VideoReader* v) : configurator(c), vidReader(v) {}

		//=========================================================================================
		void addCamera(int idx, Point pos, double previewScale, Point3d _camCoords) {
			
			int id = cameras.size();
			Camera* cam = new Camera();
			cam->id = id;
			cam->idx = idx;
			cam->viewHFlip = configurator->readObject<bool>("viewHFlip"+to_string(idx));
			cam->viewVFlip = configurator->readObject<bool>("viewVFlip"+to_string(idx));
			cam->projHFlip = configurator->readObject<bool>("projHFlip"+to_string(idx));
			cam->projVFlip = configurator->readObject<bool>("projVFlip"+to_string(idx));

			string videoName = configurator->readObject<string>("video" + to_string(idx));
			cam->capture = vidReader->addVideo(videoName);

			int vidW = int(cam->capture.get(CV_CAP_PROP_FRAME_WIDTH));
			int vidH = int(cam->capture.get(CV_CAP_PROP_FRAME_HEIGHT));
			Point previewSize (int(floor(previewScale*vidW)), int(floor(previewScale*vidH)));
			cam->viewRect = Rect(pos, pos+previewSize);
			
			cam->homography = configurator->readObject<Mat>("homography" + to_string(idx));

			vector<double> colorVctr = configurator->readObject<vector<double>>("backGrColor" + to_string(idx));
			cam->backGrColor = Scalar(colorVctr[0], colorVctr[1], colorVctr[2], colorVctr[3]);

			vector<int> ballTemplRadiusRange = configurator->readObject<vector<int>>("templRad" + to_string(idx));

			// Creates ball templates
			vector<Mat> templates = TemplateGenerator::createBallTemplVctr ( ballTemplRadiusRange[0], ballTemplRadiusRange[1], 1, cam->backGrColor );
			//reverse(templates.begin(), templates.end());
			cam->ballTemplates = templates;

			vector<int> hueIntervals = configurator->readObject<vector<int>>("hue" + to_string(idx));
			cam->hueIntervalL = hueIntervals[0];
			cam->hueIntervalR = hueIntervals[1];

			double perspectiveRatio = configurator->readObject<double>("perspectiveRatio" + to_string(idx));
			cam->perspectiveRatio = 1.15 * perspectiveRatio;

			cam->camCoords = _camCoords;

			cameras.push_back(cam);
			CAMERAS_CNT = cameras.size();
		}

		//=========================================================================================
		void updateFSize () {
			int vidW = (int) cameras[0]->capture.get(CV_CAP_PROP_FRAME_WIDTH);
			int vidH = (int) cameras[0]->capture.get(CV_CAP_PROP_FRAME_HEIGHT);
			//fSize = Point(vidW / 2, vidH / 2); -> FIX
			fSize = Point(960, 540);
		}

		//=========================================================================================
		vector<Camera*> getCameras () {
			return cameras;
		}

		//=========================================================================================
		~CameraHandler(void) {
			for (unsigned i = 0; i < cameras.size(); i++) {
				delete cameras[i];
			}
		}

};

}