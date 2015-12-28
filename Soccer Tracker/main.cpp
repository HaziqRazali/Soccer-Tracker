/************************************************************************************************* 


1 pixel in x = 0.0546m
1 pixel in y = 0.0629m

I thought Mat = pass by reference

TruePositive must contain
- String to enable / disable track
- Position of one camera
- Position of 3D ball coordinate

Tomorrow:
- Complete circularity
- If outcome is bad or not satisfying, revert to original code

Task 1 : Compile video datasets												 [ COMPLETED ]
---- ISSIA, IEEE, VSPETS

Task 2 : Multiple Camera Height Estimation									 [ COMPLETED ]
---- Write function for Triangulation										 [ DONE ]

Task 3 : Smart Tracking	+ Camera handoff									 [ COMPLETED ]
---- Find actual 3D Camera position											 [ NOT DONE ]
---- Simple spatial constraints for 2nd ball recovery	
---- Epipolar Constraint for multi-camera detection				             [ DONE ]
---- Enable feedback														 [ DONE ]
---- Camera prediction														 [ OPTIMIZE Kalman or Abandon Kalman and use Physics ]
---- trackRatio + path matching to keep cameras active for Task 6 and 4
	 ---- Pass Track Ratio to other cameras
	 ---- How to reduce Track Ratio

Task 4 : Single Camera Height Estimation									 [ COMPLETED ]
---- Allow Ball to accumulate its 3D coordinates							 [ DONE ]
---- Form equation of plane that is perpendicular to the ground plane        [ DONE ]
---- Internal Height Estimation												 [ DONE ]
---- Make it robust to outliers (RANSAC / ETC)                               [ ]
---- Forced Observation for both cameras                                     [ NOT DONE ]

Task 5 : Results and Conclusion
---- Fix Ground Truth														 [ DONE ]
---- Triangulate Ground Truth												 [ DONE ]
---- Get Results ( Precision + Recall )                                      [ DONE ]
---- Extrapolate and Plot

Task 6 : Physics Based Height Estimation
---- Establish kinematics equation


*************************************************************************************************/

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
	/*ofstream outFile[6];
	static stringstream sstm;
	static bool Switch = false;

	for (int i = 0; i < 6; i++)
	{
		sstm.str("");
		sstm << "Camera " << i + 1 << ".txt";
		outFile[i].open(sstm.str());

		for (int j = 0; j < givenTrajectories[i].size(); j++)
		{
			cout << givenTrajectories[i][j] << endl;
			outFile[i] << givenTrajectories[i][j].x << " " << givenTrajectories[i][j].y << endl;
		}
	}*/

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
