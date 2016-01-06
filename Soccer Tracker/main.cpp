/*

1 pixel in x = 0.05464m
1 pixel in y = 0.06291m
Radius of ball = 0.7m

If everything is in meters, then find out how to generate actual velocity given 2 frames

Contributions
[✓]	Compile video datasets
[✓]	3D Localization
[✓]	Multiple and Single Camera Height Estimation
[]	Physics Based Height Estimation
[]	Camera Handoff
[]	Camera Coordinates Extraction

[✓]	Results and Conclusion

Others
[]	Ball recovery when it goes 'out' of field
[]  Write about 'administration' in report ie. placement of camera, conversion of pixel (modelView) to meters

Camera Handoff
[✓] Revert to original
[✓] Internal Height Estimation will form plane
[✓] Find out how to store Mat planeTrajectory
[✓] Plane will remain until double detections are found
- Physics will establish trajectory
- Determine landing spot for fun
- All detections will be checked against plane

Use mat.copyTo to clear region

*/

#include <opencv\cv.h>
#include <opencv2\opencv.hpp>

#include "xmlParser.h"
#include "TrajectoryAnalyzer.h"
#include "Configurator.h"
#include "VideoReader.h"
#include "videoWriter.h"
#include "CameraHandler.h"
#include "TrackInfo.h"
#include "BackGroundRemover.h"
#include "ContourAnalyzer.h"
#include "Tracker.h"
#include "MultiCameraTracker.h"
#include "globalSettings.h"

#include "omp.h"

using namespace cv;
using namespace std;
using namespace st;

//*************************************************************************************************
// ----- This is a main source file it handles the whole process of tracking:
// ----- opens video, reads all configuration and calibration data, starts tracking,
// ----- manipulate threads, handles keyboard and mouse control, etc.
//*************************************************************************************************


int x_, y_, TID_mouseUpdate = -1;

double scalePreview;
int pauseFlag = 0;
vector<Rect> cameraViewRects;
int globalFrameCount = 0;


//=================================================================================================
void _onMouse(int event, int x, int y, int flags, void* param) {

	if (event == CV_EVENT_LBUTTONUP) 
	{
		#pragma omp flush(pauseFlag, TID_mouseUpdate)
		if ((pauseFlag == 2) && (TID_mouseUpdate == -1)) 
		{
			Point clickP = Point(x, y);
			for (unsigned i = 0; i < cameraViewRects.size(); i++) 
			{
				Rect r = cameraViewRects[i];
				if (r.contains(clickP)) 
				{
					clickP -= Point(r.x, r.y);
					#pragma omp critical(x_)
					x_ = int(1.0 / (scalePreview / scaleLoad) * clickP.x);
					#pragma omp critical(y_)
					y_ = int(1.0 / (scalePreview / scaleLoad) * clickP.y);
					#pragma omp critical(TID_mouseUpdate)
					TID_mouseUpdate = int(i);
					break;
				}
			}
		}
	}

	else if (event == CV_EVENT_RBUTTONDOWN)	{
		// ---------- PAUSE mode ----------
		#pragma omp critical(pauseFlag)
		{
			if (pauseFlag != 2) 
			{
				printf("PAUSE\n"); fflush(stdout);
				pauseFlag = 2;
			}
			else if (pauseFlag == 2) 
			{
				printf("CONTINUE\n"); fflush(stdout);
				pauseFlag = 0;
			}
		}
	}
}


//=================================================================================================
int main() {
	
	vector<vector<Point>> givenTrajectories;
	#ifdef NOT_FROM_THE_BEGINING
	TrajectoryAnalyzer::readFullTrajectory(givenTrajectories, 0.5, START_FRAME, 2997);
	#else
	TrajectoryAnalyzer::readFullTrajectory(givenTrajectories, 0.5, 0, 2997);
	#endif

	ofstream outFile[6];
	stringstream sstm;

	for (int i = 0; i < 6; i++)
	{
		sstm.str("");
		sstm << "Camera " << i + 1 << ".txt";
		outFile[i].open(sstm.str());
	}
	
	// Save Ground Truth to file
	//ofstream outFile[6];
	//static stringstream sstm;

	//for (int i = 0; i < 6; i++)
	//{
	//	sstm.str("");
	//	sstm << "Camera " << i + 1 << ".txt";
	//	outFile[i].open(sstm.str());

	//	for (int j = 0; j < givenTrajectories[i].size(); j++)
	//	{
	//		//cout << givenTrajectories[i][j] << endl;
	//		outFile[i] << givenTrajectories[i][j].x << " " << givenTrajectories[i][j].y << endl;
	//	}
	//}

	//return 0;

	//Ground truth visualization
	/*for (unsigned i = 0; i < givenTrajectories.size(); i++) 
	{
		cout << givenTrajectories[i].size();
		Mat trajM(540, 960, CV_8UC3, CV_RGB(255,255,255));
		TrajectoryAnalyzer::drawTrajectory(trajM, givenTrajectories[i]);
		resize(trajM, trajM, Size(), 0.5, 0.5, INTER_AREA);
		imshow("tr" + to_string(i), trajM);
	}
	while (true) {
		if (waitKey(1) == 27) {
			break;
		} 
	}
	return 0;*/

	// ----- create configurator to parse xml file with settings -----
	Configurator* configurator = new Configurator("config.xml");
	// ----- create videoReader to read video from all cameras -----
	st::VideoReader* videoReader = new VideoReader();
	#ifdef WRITE_VIDEO
	// ----- create video writer -----
	string outputVideoFolder = configurator->readObject<string>("outputVideoFolder") + getTimeString() + "\\";
	createFolder(outputVideoFolder);
	st::VideoWriter videoWriter(outputVideoFolder);

	#endif

	CameraHandler camHandler(configurator, videoReader);
	MultiCameraTracker mcTracker;

	
	//[Col, Row]
	//Camera 3 coord	= [960, 1080]
	//		   Ball = [578, 64] ->[1054, 77]
	//				= [559, 57] ->[1038, 45]
	//				= [540, 50] ->[1022, 12]

	//[Col, Row]
	//Camera 4 coord	= [960, 0]
	//		   Ball = [295, 386] ->[1051, 171]
	//				= [327, 373] ->[1036, 191]
	//				= [357, 361] ->[1022, 210]
	//

	////Debug - Need to project ground truth
	//pair<double, Point3d> V1 = mcTracker.Triangulate(Point3d(960, 1080, 140), Point3d(1054, 77, 0), Point3d(960, 0, 140), Point3d(1051, 171, 0));
	//pair<double, Point3d> V2 = mcTracker.Triangulate(Point3d(960, 1080, 140), Point3d(1038, 45, 0), Point3d(960, 0, 140), Point3d(1036, 191, 0));
	//pair<double, Point3d> V3 = mcTracker.Triangulate(Point3d(960, 1080, 140), Point3d(1022, 12, 0), Point3d(960, 0, 140), Point3d(1022, 210, 0));
	////cout << V1.second << endl << V2.second << endl << V3.second << endl;

	//vector<Point3d> V = { V1.second, V2.second, V3.second };
	//Mat plane = mcTracker.formPlane(V);
	//cout << plane << endl;

	//return 0;

	clock_t tic = clock();

	Mat cameraView(gui_camPreviewH, gui_camPreviewW, CV_8UC3);
	scalePreview = 1.0 / 3; // preview size will be 640 x 360

	// Camera Coordinates are not at the extreme corners. Origin is at top left
	camHandler.addCamera(1, Point(1280, 540), scalePreview, Point3d(1620, 1080, 140));	// Original: 0, 540		
	camHandler.addCamera(2, Point(1280, 180), scalePreview, Point3d(1620, 0, 140));		// Original: 640, 540	
	camHandler.addCamera(3, Point(640, 540), scalePreview, Point3d(960, 1080, 140));	// Original: 1280, 540

	camHandler.addCamera(4, Point(640, 180), scalePreview, Point3d(960, 0, 140));		// Original: 0, 180		
	camHandler.addCamera(5, Point(0, 540), scalePreview, Point3d(300, 1080, 140));		// Original: 640, 180
	camHandler.addCamera(6, Point(0, 180), scalePreview, Point3d(300, 0, 140));			// Original: 1280, 180

	camHandler.updateFSize();
	vector<Camera*> allCameras = camHandler.getCameras();

	#ifdef WRITE_VIDEO
	// ---------- create output videos ----------
	int vidOutExt = CV_FOURCC('M', 'J', 'P', 'G'); //videoReader->getCodecExt();	// another way: vidOutExt = CV_FOURCC('M','J','P','G');
	for (auto camera : allCameras) {
		videoWriter.addVideo(to_string(camera->idx) + "_.avi", fSize, OUT_FRAME_RATE, vidOutExt, camera->idx);
	}
	videoWriter.addVideo("all_.avi", Size(gui_camPreviewW, gui_camPreviewH), OUT_FRAME_RATE, vidOutExt, -1);
	videoWriter.addVideo("model_.avi", Size(gui_modelW, gui_modelH), OUT_FRAME_RATE, vidOutExt, -2);
	#endif

	// ---------- prepare for multithreading ----------
	omp_set_num_threads(CAMERAS_CNT + 1);
	int processedFrames_s = 0, camerasReady = 0;
	TrackInfo* trackInfo = new TrackInfo[CAMERAS_CNT];
	bool showTraj = false, slowMotion = false;

	ID_counter = 0;
	ID_groups_cnt = 10;

	//*********************************************************************************************
	//******************************** parallel threads *******************************************
	//*********************************************************************************************
	#pragma omp parallel shared(trackInfo, processedFrames_s, x_, y_, camerasReady, pauseFlag, cameraView, showTraj, slowMotion)
	{
		int TID = omp_get_thread_num();

		int debugger = 0;

		#pragma omp critical
		trackInfo[TID].allowTracking = true;
		ID_shift = TID;

		if (TID < CAMERAS_CNT) 
		{
			//_____________________________________________________________________________________
			//********************************************* camera thread *************************
			//*************************************************************************************
			int processedFrames = 0;
			// ----- do some initialization for every camera -----
			Camera* camera = allCameras[TID];
			VideoCapture camCapture = camera->capture;
			Rect camViewRect = camera->viewRect;
			bool horFlip = camera->viewHFlip;
			#ifdef WRITE_VIDEO
			cv::VideoWriter vidWriter = videoWriter.getVideoWriter(camera->idx);
			#endif
			Mat frame;
			BackGroundRemover remover(500, 256, 5, false);
			ContourAnalyzer cAnalyzer;

			Tracker tracker;
			tracker.initialize();
			tracker.setBallTempls(camera->ballTemplates);
			tracker.setPerspectiveRatio(camera->perspectiveRatio);
			tracker.setGivenTrajectory(givenTrajectories[TID]);
			//=====================================================

			while (true) {
				// ========== check value of pauseFlag ==========
				#pragma omp flush (pauseFlag)
				if (pauseFlag == 2) 
				{
					// ----- switch to PAUSE mode -----
					waitKey(10);
					#pragma omp flush(TID_mouseUpdate, x_, y_)
					if (TID_mouseUpdate == TID) 
					{
						tracker.ball_addCandidateManually(x_, y_);
						#pragma omp critical
						TID_mouseUpdate = -1;
					}

					else 
					{
						continue;
					}
				}
				else if (pauseFlag == 1) 
				{
					// ----- STOP all threads -----
					#pragma omp critical
					processedFrames_s = processedFrames;
					vector<pair<int, Point>> traj;
					printf("thread %d processed %d frames\n", TID, processedFrames); fflush(stdout);
					break;
				}

				// ========== retrieve new frame ==========
				if (!camCapture.read(frame)) 
				{
					#pragma omp critical
					pauseFlag = 1;
					continue;
				};
				
				#ifdef NOT_FROM_THE_BEGINING
				if (processedFrames < START_FRAME) 
				{
					processedFrames++;
					continue;
				}
				#endif
		
				// ========== preprocess frame ==========
				resize(frame, frame, Size(960,540), 0, 0, INTER_AREA);

				if (horFlip) { flip(frame, frame, 1); } // !!!!! some source videos might be flipped !!!!!

				Mat mask = remover.processFrame(frame);

				vector<Rect> players_cand;
				vector<Point> ball_cand;

				cAnalyzer.process(mask, players_cand, ball_cand);
				// ========== wait for permission to start tracking ==========
				bool _flg = true;

				while (_flg) 
				{
					// !!!!! #omp: do not use flush with pointers! use critical section instead !!!!!
					#pragma omp critical
					if (trackInfo[TID].allowTracking || (pauseFlag == 1)) _flg = false;
				}

				#pragma omp critical(trackInfo)

				/********************************************************************************
											First Stage of Analysis
				*********************************************************************************/
				
				tracker.setTrackInfo(trackInfo[TID]);
				tracker.processFrame(frame, ball_cand, players_cand, mcTracker.getTruePositives(), TID, processedFrames, outFile[TID]);
				trackInfo[TID] = tracker.getTrackInfo();

				#pragma omp critical
				{
					trackInfo[TID].allowTracking = false;
					camerasReady++;
				}

				// ========== display results for single camera ==========
				#pragma omp flush(showTraj)
				if (showTraj)
				{
					tracker.getTrajFrame(frame);
				}
				else 			
				{
					tracker.drawTrackingMarks(frame, TID);
				}
				
				#ifdef WRITE_VIDEO
				vidWriter << frame;
				#endif

				resize(frame, frame, Size(camViewRect.width, camViewRect.height));
				frame.copyTo(cameraView(camViewRect));

				if (TID == 0) 
				{
					cameraView(Rect(0, 0, 200, 100)) = CV_RGB(0, 0, 0);
					putText(cameraView, to_string(processedFrames), Point(10, 50), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255));
				}


				globalFrameCount = processedFrames;
				processedFrames++;
			}
		}

		else 
		{
			//_____________________________________________________________________________________
			//********************************************* handler thread ************************
			//*************************************************************************************

			// ========== do initialization ==========
			mcTracker      = MultiCameraTracker();
			Mat model	   = imread(configurator->readObject<string>("fieldModel"));

			mcTracker.setFieldModel(model);
			mcTracker.setCameras(allCameras);

			namedWindow("modelView", CV_WINDOW_AUTOSIZE);
			namedWindow("cameraView", CV_WINDOW_AUTOSIZE);
			setMouseCallback("cameraView", _onMouse);

			#ifdef WRITE_VIDEO
			cv::VideoWriter vidWriter_all = videoWriter.getVideoWriter(-1);
			cv::VideoWriter vidWriter_model = videoWriter.getVideoWriter(-2);
			#endif

			Mat modelPreview;

			// ========== loop for all frames ==========
			while (true) 
			{
				// ---------- STOP mode ----------
				#pragma omp flush(pauseFlag)
				if (pauseFlag == 1) 
				{
					printf("handler thread stopped\n"); fflush(stdout);
					while (true) if (waitKey(1) == 'q')	break;
					destroyAllWindows();
					break;
				}

				#pragma omp flush(camerasReady)
				if (camerasReady == CAMERAS_CNT) 
				{
					#pragma omp critical

					/********************************************************************************
											Second (Final) Stage of Analysis
					*********************************************************************************/
					
					mcTracker.updateTrackData(trackInfo);
					mcTracker.process();
					mcTracker.finalizeResults(modelPreview, cameraView);

					imshow("modelView", modelPreview);
					imshow("cameraView", cameraView);

					#ifdef WRITE_VIDEO
					resize(modelPreview, modelPreview, Size(gui_modelW, gui_modelH));
					vidWriter_model << modelPreview;
					vidWriter_all << cameraView;
					#endif

					// ---------- cameras allowed to process next frame ----------
					#pragma omp critical
					camerasReady = 0;
					#pragma omp critical
					{
						for (int i = 0; i < CAMERAS_CNT; i++)
							trackInfo[i].allowTracking = true;
					}
				}
				// ========== wait for any key to be pressed ======================================
				for (auto i = 0; i < SLOW_MOTION_REPEAT_TIME; i++) 
				{
					switch (waitKey(1)) 
					{
						//_________________________________________________________________________
					case 27: {	//---------------------------------------- STOP mode ----------
						#pragma omp critical
						pauseFlag = 1;
						break;
					}
							 //_________________________________________________________________________
					case 't': { //--------------------------------- TRAJECTORY MODE ----------
						#pragma omp critical
						showTraj = !showTraj;
						break;
					}
							  //_________________________________________________________________________
					case ' ': { //-------------------------------------- PAUSE mode ----------
						#pragma omp critical
							{
								if (pauseFlag != 2) {
									printf("PAUSE\n"); fflush(stdout);
									pauseFlag = 2;
								}
								else if (pauseFlag == 2) {
									printf("CONTINUE\n"); fflush(stdout);
									pauseFlag = 0;
								}
							}
							break;
					}
							  //_________________________________________________________________________
					case 's': { //-------------------------------- SLOW MOTION mode ----------
						#pragma omp critical
						slowMotion = !slowMotion;
						break;
					}
					}

					if (!slowMotion) {
						break;
					}
				}
				//=================================================================================
			}

		}
	}

	double allTime = double((clock() - tic)) / CLOCKS_PER_SEC;
	double fps = double(processedFrames_s) / allTime;
	printf("finished in %f seconds\n%f fps\n", allTime, fps);
	delete[] trackInfo;
	delete videoReader;
	delete configurator;

	system("PAUSE");
	return 0;
}

// ********** example of code to save backGrColor to file **********
//Configurator configurator("config.xml");
//for (int i = 1; i <= 6; i++) {
//	cout << "camera" + to_string(i) << endl;
//	Scalar meanColor = getMeanBackGrColor("c:\\SoccerVideos\\static2\\" + to_string(i) + ".avi", 500);
//	configurator.writeObject<Scalar>("meanBackGrColor_cam" + to_string(i), meanColor);
//}
//cout << "done" << endl;
