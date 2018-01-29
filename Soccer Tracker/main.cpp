#include <opencv2/opencv.hpp>
#include <opencv/cv.h>

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

void testFunc();

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

	ofstream t_Error;
	stringstream _sstm;

	_sstm.str("");
	_sstm << "TriangulationError" << ".txt";
	t_Error.open(_sstm.str());

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

	clock_t tic = clock();

	// Initialize camera + labels
	Mat cameraView(gui_camPreviewH, gui_camPreviewW, CV_8UC3);
	putText(cameraView, "Camera 5", Point(0    + 230, 930), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0));
	putText(cameraView, "Camera 3", Point(640  + 230, 930), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0));
	putText(cameraView, "Camera 1", Point(1280 + 230, 930), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0));

	putText(cameraView, "Camera 6", Point(0    + 230, 170), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0));
	putText(cameraView, "Camera 4", Point(640  + 230, 170), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0));
	putText(cameraView, "Camera 2", Point(1280 + 230, 170), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0));

	scalePreview = 1.0 / 3; // preview size will be 640 x 360

	// Camera Coordinates are not at the extreme corners. Origin is at top left
	camHandler.addCamera(1, Point(1280, 540), scalePreview, Point3d(87.56, 67.928 + 34.52, 60.69));	// Original: 0, 540		
	camHandler.addCamera(2, Point(1280, 180), scalePreview, Point3d(87.69, 67.928 - 103.14, 60.62));		// Original: 640, 540

	camHandler.addCamera(3, Point(640, 540), scalePreview, Point3d(53.15, 67.928 + 34.13, 56.51));	// Original: 1280, 540
	camHandler.addCamera(4, Point(640, 180), scalePreview, Point3d(52.37, 67.928 - 101.78, 57.93));		// Original: 0, 180		

	camHandler.addCamera(5, Point(0, 540), scalePreview, Point3d(27.84, 67.928 + 33.97, 59.50));		// Original: 640, 180
	camHandler.addCamera(6, Point(0, 180), scalePreview, Point3d(27.84, 67.928 - 102.09, 58.83));			// Original: 1280, 180

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
		
		Ptr<BackgroundSubtractorMOG2> MOG2;
		MOG2 = createBackgroundSubtractorMOG2();
		MOG2->setShadowValue(0);

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
			tracker.initialize(TID);
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

				Mat mask;
				mask = remover.processFrame(frame, TID);

				/*if (TID == 3)
				{
					char filename[40];
					sprintf(filename, "result.png");
					imwrite(filename, mask);
					waitKey(0);
				}*/

				/*MOG2->apply(frame, mask);

				erode(mask, mask, getStructuringElement(MORPH_RECT, Size(3, 3)));
				dilate(mask, mask, getStructuringElement(MORPH_RECT, Size(3, 3)));*/
				
				Mat playerMask = mask.clone();

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
				tracker.processFrame(frame, ball_cand, players_cand, mcTracker.getTruePositives(), TID, processedFrames, outFile[TID], playerMask);
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
					cameraView(Rect(640 + 210, 0, 200, 100)) = CV_RGB(0, 0, 0);
					putText(cameraView, to_string(processedFrames), Point(640 + 280, 50), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255));
				}


				globalFrameCount = processedFrames;
				processedFrames++;
			}
		}

		/*#ifndef THREE_DIMENSIONAL_ANALYSIS
		else
		{
			namedWindow("cameraView", CV_WINDOW_AUTOSIZE);
			imshow("cameraView", cameraView);

			#pragma omp critical
			camerasReady = 0;
			#pragma omp critical
			{
				for (int i = 0; i < CAMERAS_CNT; i++)
					trackInfo[i].allowTracking = true;
			}
		}
		#endif*/

		#ifdef THREE_DIMENSIONAL_ANALYSIS
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

			namedWindow("modelView", CV_WINDOW_NORMAL);
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
					mcTracker.process(t_Error, globalFrameCount);
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
		#endif
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

void testFunc2() {



}

void testFunc() {

	BackGroundRemover remover(500, 256, 5, false);

	int hbins = 18;
	int channels[] = { 0 };
	int histSize[] = { hbins };
	float hranges[] = { 0, 180 };
	const float* ranges[] = { hranges };

	Mat patch_HSV_A, patch_HSV_B, patch_HSV_C;
	Mat imgA_mask, imgB_mask, imgC_mask;
	MatND HistA, HistB, HistC;

	// Read in RGB
	Mat imgA = imread("White1.png");
	Mat imgB = imread("White3.png");
	Mat imgC = imread("Blue1.png");

	// Convert to HSV
	cvtColor(imgA, patch_HSV_A, CV_BGR2HSV);
	cvtColor(imgB, patch_HSV_B, CV_BGR2HSV);
	cvtColor(imgC, patch_HSV_C, CV_BGR2HSV);

	// Generate mask ( remove green pixels )
	remover.HistMethod(imgA, imgA_mask);
	remover.HistMethod(imgB, imgB_mask);
	remover.HistMethod(imgC, imgC_mask);

	//// Extract Hue
	//vector<Mat> hsv_planes;
	//split(patch_HSV_A, hsv_planes);

	//// Remove green pixels from HUE
	//Mat mask = imgA_mask / 255;
	//cout << mask;
	//multiply(hsv_planes[0], mask, hsv_planes[0]);

	namedWindow("Player A", CV_WINDOW_NORMAL); imshow("Player A", imgA);	//namedWindow("Mask A", CV_WINDOW_NORMAL);  imshow("Mask A", imgA_mask);
	namedWindow("Player B", CV_WINDOW_NORMAL); imshow("Player B", imgB);	//imshow("Mask B", imgB_mask);
	namedWindow("Player C", CV_WINDOW_NORMAL); imshow("Player C", imgC);	imshow("Mask C", imgC_mask);

	// Calculate histogram and normalize
	calcHist(&patch_HSV_A, 1, channels, imgA_mask, HistA, 1, histSize, ranges, true, false);
	normalize(HistA, HistA, 0, 255, CV_MINMAX);

	calcHist(&patch_HSV_B, 1, channels, imgB_mask, HistB, 1, histSize, ranges, true, false);
	normalize(HistB, HistB, 0, 255, CV_MINMAX);

	calcHist(&patch_HSV_C, 1, channels, imgC_mask, HistC, 1, histSize, ranges, true, false);
	normalize(HistC, HistC, 0, 255, CV_MINMAX);

	// COMPARE SIMILARITY
	cout << "A vs B = " << compareHist(HistA, HistB, CV_COMP_BHATTACHARYYA) << endl;
	cout << "A vs C = " << compareHist(HistA, HistC, CV_COMP_BHATTACHARYYA) << endl;
	cout << "B vs C = " << compareHist(HistB, HistC, CV_COMP_BHATTACHARYYA) << endl;
	
	//Mat for drawing 
	Mat histimg = Mat::zeros(200, 320, CV_8UC3);
	histimg = Scalar::all(0);
	int binW = histimg.cols / hbins;
	Mat buf(1, hbins, CV_8UC3);

	//Set RGB color
	for (int i = 0; i < hbins; i++)	buf.at< Vec3b>(i) = Vec3b(saturate_cast< uchar>(i*180. / hbins), 255, 255);

	cvtColor(buf, buf, CV_HSV2BGR);

	//drawing routine
	for (int i = 0; i < hbins; i++)
	{
		// UPDATE HERE !
		int val = saturate_cast< int>(HistA.at< float>(i)*histimg.rows / 255);

		rectangle(histimg, Point(i*binW, histimg.rows),
			Point((i + 1)*binW, histimg.rows - val),
			Scalar(buf.at< Vec3b>(i)), -1, 8);
		int r, g, b;
		b = buf.at< Vec3b>(i)[0];
		g = buf.at< Vec3b>(i)[1];
		r = buf.at< Vec3b>(i)[2];

		//show bin and RGB value
		//printf("[%d] r=%d, g=%d, b=%d , bins = %d \n", i, r, g, b, val);
	}
	imshow("Histogram", histimg);
}

// ********** example of code to save backGrColor to file **********
//Configurator configurator("config.xml");
//for (int i = 1; i <= 6; i++) {
//	cout << "camera" + to_string(i) << endl;
//	Scalar meanColor = getMeanBackGrColor("c:\\SoccerVideos\\static2\\" + to_string(i) + ".avi", 500);
//	configurator.writeObject<Scalar>("meanBackGrColor_cam" + to_string(i), meanColor);
//}
//cout << "done" << endl;
