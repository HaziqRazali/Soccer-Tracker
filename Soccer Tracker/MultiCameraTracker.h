#pragma once

#include "CameraHandler.h"
#include "globalSettings.h"
#include "KalmanFilter.h"
#include "TrackInfo.h"
#include "Tracker.h"

#include <map>
#include <utility>
#include <limits>
#include <xutility>
#include <algorithm>
#include <fstream>
#include <time.h>

namespace st {

//*************************************************************************************************
// ----- struct describing a rectangle projected to the field model
//*************************************************************************************************
struct Polygon {

	vector<Point2f> points;

	//=============================================================================================
	Polygon () {}

	//=============================================================================================
	Polygon (Rect& rect, Mat& homography) {
		vector<Point2f> tvctr(4);
		tvctr[0] = Point2f(float(rect.x), float(rect.y));
		tvctr[1] = Point2f(float(rect.x + rect.width), float(rect.y));
		tvctr[2] = Point2f(float(rect.x + rect.width), float(rect.y + rect.height));
		tvctr[3] = Point2f(float(rect.x), float(rect.y + rect.height));
		perspectiveTransform(tvctr, points, homography);
	}

	//=============================================================================================
	void flipVert (int height) {
		for (auto& p : points) {
			p.y = height - p.y;
		}
	}
};

//*************************************************************************************************
// ----- struct describing a ball candidate projected to the field model
//*************************************************************************************************
struct ProjCandidate {

	// Camera Info _____________________________________________________________
	int cameraID;					  // Camera ID
	Point3d camCoords;				  // 3D camera coordinates in meters
	Mat homography;					  // Camera homography matrix
	
	// Ball Info _____________________________________________________________
	int ID, ID2;				      // Ball ID  
	int teamID = 3;

	vector<Point2f> coords, coordsKF; // 2D projected coordinates
	vector<Point2f> coords_meters;    // 2D projected coordinates in meters
	Point coords_pred;			      // 2D projected predicted coordinates
	vector<Point3d> coords3D;		  // 3D coordinates of Ball	in meters	 [ multiple objects will store the same coordinates ]

	pair<Point,int> other_coord;	  // <2D unprojected coordinates of ball, wrt. ID of opposite camera>
	bool isRealBall;                  // True Positive
	
	// Ground Truth _____________________________________________________________
	Point GTcoord = Point(-1,-1);	  // 2D unprojected ground truth coordinates. (-1,-1) if does not exist
	Point3d GTcoords3D;               // 3D true coordinates of Ball		 [ multiple objects will store the same coordinates ]
	vector<Camera*> cameraVisible;    // Cameras that Triangulated the ball  [ multiple objects will store the same ID ]

	// Others _____________________________________________________________
	vector<Polygon> bounds;
	vector<int> frames;
	st::KalmanFilter KF;
		
	double trackRatio;

	//=============================================================================================
	ProjCandidate (int _ID = -1, int _cameraID = -1, Point3d _camCoords = Point3d(), Mat _homography = Mat()) : ID(_ID), cameraID(_cameraID), camCoords(_camCoords), homography(_homography), isRealBall(false) {
		// processNoiseCov, measureNoiseCov, errorCov
		KF = st::KalmanFilter(0.001, 0.1, 0.01);
	}

	//=============================================================================================
	void update (TrackInfo& ti, int frameCnt, bool flipH) {

		vector<Point2f> p_orig = vector<Point2f>(1);
		vector<Point2f> p_proj = vector<Point2f>(1);

		Point2f _coord;
		Polygon _bound;

		/********************************************************************************
				Projects the measured 2D coordinates of mainCandidate ( frame t )
		*********************************************************************************/
		
		p_orig[0] = ti.coord;
		
		// Upscale coordinates for projection onto top View ( 1920 x 1080 )
		p_orig[0].x = p_orig[0].x * 2;
		p_orig[0].y = p_orig[0].y * 2;

		// If vertical flip
		if (flipH) p_orig[0].x = 1920 - p_orig[0].x;

		// Projection function
		perspectiveTransform(p_orig, p_proj, homography);

		_coord = p_proj[0];
		_bound = Polygon(ti.rect, homography);

		Point2f _coordsKF = KF.process(_coord);

		// Push back coordinate (pixel)
		coords.push_back(_coord);
		coordsKF.push_back(_coordsKF);

		// Push back coordinate (convert pixel to meters)
		coords_meters.push_back(Point2f(_coordsKF.x * 0.05464, _coordsKF.y * 0.06291));

		/********************************************************************************
				Projects the predicted 2D coordinates of mainCandidate ( frame t + 1 ) Not in use
		*********************************************************************************/
		
		p_orig[0] = ti.predCoord;

		// Upscale coordinates for projection onto top View ( 1920 x 1080 )
		p_orig[0].x = p_orig[0].x * 2;
		p_orig[0].y = p_orig[0].y * 2;

		// Projection function
		perspectiveTransform(p_orig, p_proj, homography);

		_coord = p_proj[0];
		coords_pred = _coord;

		/********************************************************************************
						Projects the Ground Truth coordinates ( frame t )
		*********************************************************************************/

		if (ti.GTcoord != Point(-1, -1))
		{
			p_orig[0] = ti.GTcoord;

			// Upscale coordinates for projection onto top View ( 1920 x 1080 )
			p_orig[0].x = p_orig[0].x * 2;
			p_orig[0].y = p_orig[0].y * 2;

			// Projection function
			perspectiveTransform(p_orig, p_proj, homography);

			_coord = p_proj[0];
			GTcoord = _coord;
		}

		/********************************************************************************
											Others
		*********************************************************************************/

		//trackRatio = ti.trackRatio;
		bounds.push_back(_bound);
		frames.push_back(frameCnt);

	}

	void updatePlayer(PlayerCandidate* ti, int frameCnt, bool flipH) {

		vector<Point2f> p_orig = vector<Point2f>(1);
		vector<Point2f> p_proj = vector<Point2f>(1);

		Point2f _coord;
		Polygon _bound;

		/********************************************************************************
				Projects the measured 2D coordinates of playerCandidate ( frame t )
		*********************************************************************************/

		teamID = ti->teamID;

		p_orig[0] = ti->curCrd;

		// Upscale coordinates for projection onto top View ( 1920 x 1080 )
		p_orig[0].x = p_orig[0].x * 2;
		p_orig[0].y = p_orig[0].y * 2;

		// If vertical flip
		if (flipH) p_orig[0].x = 1920 - p_orig[0].x;

		// Projection function
		perspectiveTransform(p_orig, p_proj, homography);

		_coord = p_proj[0];
		_bound = Polygon(ti->curRect, homography);

		Point2f _coordsKF = KF.process(_coord);

		// Push back coordinate (pixel)
		coords.push_back(_coord);
		coordsKF.push_back(_coordsKF);

		// Push back coordinate (convert pixel to meters)
		coords_meters.push_back(Point2f(_coordsKF.x * 0.05464, _coordsKF.y * 0.06291));

		/********************************************************************************
		Others
		*********************************************************************************/

		//trackRatio = ti.trackRatio;
		bounds.push_back(_bound);
		frames.push_back(frameCnt);


	}

	//=============================================================================================
	Point2f getLastPoint () {
		return *(coords.end()-1);
	}

	//=============================================================================================
	Polygon getLastPolygon () {
		return *(bounds.end()-1);
	}

};

//*************************************************************************************************
// ----- This class combines and analyses the information from all cameras,
// ----- makes the projections and calculates the projected ball position
//*************************************************************************************************
class MultiCameraTracker {

	//_____________________________________________________________________________________________
	private:

		int framesProcessed;
		Mat fieldModel, Ball;

		vector<Camera*> cameras;
		vector<pair<Polygon, int>> marks;

		vector<Scalar> colors;
		vector<Scalar> teamColors;

		vector<ProjCandidate*> currCandidates;
		vector<ProjCandidate*> currPlayerCandidates[6];
		vector<ProjCandidate*> updatedPlayerCand;

		vector<vector<pair<Point,Point>>> result_final = vector<vector<pair<Point,Point>>>(6);

		vector<Point3d> finalCoords;

		Point3d finalPoint3D;
		Point3d GTfinalPoint3D;

		/*********************************************************
							Physics
		**********************************************************/
		Mat planeTrajectory;
		Point3d velocity;
		float landingTime;
		vector<Point3d> estimatedTrajectory;
		Point landingPos;

		/*********************************************************
							Object handoff
		**********************************************************/
		Rect transitRegion_A = Rect(Point(64, 0), Point(76, 68)); // Handover from 1-2
		Rect transitRegion_B = Rect(Point(28, 0), Point(50, 68));   // Handover from 2-3 (40,68)
				
		int STATE = 0;

		//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		MultiCameraTracker (void) {
			framesProcessed = 0;

			colors.push_back(CV_RGB(128,0,0));
			colors.push_back(CV_RGB(0,128,0));
			colors.push_back(CV_RGB(0,0,128));
			colors.push_back(CV_RGB(128,128,0));
			colors.push_back(CV_RGB(128,0,128));
			colors.push_back(CV_RGB(0,128,128));

			teamColors.push_back(CV_RGB(255, 255, 255));
			teamColors.push_back(CV_RGB(0, 0, 255));
			teamColors.push_back(CV_RGB(0, 0, 0));
			teamColors.push_back(CV_RGB(128,50,0));
		}

		//=========================================================================================
		void setFieldModel (Mat& model, Mat& ball = Mat()) {
			fieldModel = model;
			Ball = ball;
		}
		
		//=========================================================================================
		void setCameras (vector<Camera*>& cameras) {
			this->cameras = cameras;
		}

		//=========================================================================================
		void updateTrackData (TrackInfo trackInfo[]) {

			//#ifdef DISPLAY_GROUND_TRUTH
			//GroundTruth = vector<ProjCandidate*>();
			//for (int i = 0; i < CAMERAS_CNT; i++)
			//{
			//	// if GT exist for current camera
			//	if (trackInfo[i].GTcoord != Point(-1, -1))
			//	{
			//		ProjCandidate* newCand = new ProjCandidate(-1, -1, cameras[i]->camCoords, cameras[i]->homography);
			//		newCand->update(trackInfo[i], framesProcessed, cameras[i]->projVFlip);
			//		GroundTruth.push_back(newCand);
			//	}
			//}
			//#endif
			
			/********************************************************************************
										Prepare for Multi Camera Analysis (Ball)
			*********************************************************************************/

			vector<ProjCandidate*> used;
			for (int i = 0; i < CAMERAS_CNT; i++) 
			{
				// ID of ball
				int candidateId = trackInfo[i].ballCandID;
				if (candidateId == -1) { continue; }

				bool exists = false;

				// Update (Only if currCandidates exists)
				for (auto cc : currCandidates)
				{
					if (cc->ID == candidateId) 
					{
						used.push_back(cc);
						cc->update(trackInfo[i], framesProcessed, cameras[i]->projHFlip);
						exists = true;
						break;
					}
				}
				
				// Create new currCandidates if no similar candidates exist
				if (!exists) 
				{
					ProjCandidate* newCand = new ProjCandidate(candidateId, i, cameras[i]->camCoords, cameras[i]->homography);

					newCand->update(trackInfo[i], framesProcessed, cameras[i]->projHFlip);
					currCandidates.push_back(newCand);
					used.push_back(newCand);
				}
			}

			// --- delete currCandidate that is not updated (meaning lost)
			vector<ProjCandidate*> toDelete;

			for (auto c : currCandidates) 
			{
				if (find(used.begin(), used.end(), c) == used.end()) 
				{
					toDelete.push_back(c);
				}
			}

			for (auto& obj : toDelete) 
			{
				currCandidates.erase(remove(currCandidates.begin(), currCandidates.end(), obj), currCandidates.end());
				delete obj;
			}

			toDelete.clear();

			/********************************************************************************
									Prepare for Multi Camera Analysis (Player)
			*********************************************************************************/

			// For each camera view
			for (int i = 0; i < CAMERAS_CNT; i++)
			{
				vector<ProjCandidate*> used;

				// Loop through all candidates of camera i
				for (auto p : trackInfo[i].pCandidates)
				{
					// ID of player
					int candidateID = p->id;					
					bool exists = false;

					// Update (Only if player_currCandidates exists)
					for (auto pc : currPlayerCandidates[i])
					{
						if (pc->ID == candidateID)
						{
							used.push_back(pc);
							pc->updatePlayer(p, framesProcessed, cameras[i]->projHFlip);
							exists = true;
							break;
						}
					}

					// Create new currCandidates if no similar candidates exist
					if (!exists)
					{
						ProjCandidate* newCand = new ProjCandidate(candidateID, i, cameras[i]->camCoords, cameras[i]->homography);

						newCand->updatePlayer(p, framesProcessed, cameras[i]->projHFlip);
						currPlayerCandidates[i].push_back(newCand);
						used.push_back(newCand);
					}
				}

				// --- delete currCandidate that is not updated (meaning lost)
				vector<ProjCandidate*> toDelete;

				for (auto pc : currPlayerCandidates[i])
				{
					if (find(used.begin(), used.end(), pc) == used.end())
					{
						toDelete.push_back(pc);
					}
				}

				for (auto& obj : toDelete)
				{
					currPlayerCandidates[i].erase(remove(currPlayerCandidates[i].begin(), currPlayerCandidates[i].end(), obj), currPlayerCandidates[i].end());
					delete obj;
				}

				toDelete.clear();

			}

			framesProcessed++;
		}

		//=========================================================================================
		void process(ofstream& file, int globalFrameCount) {
			
			// True positive identification
			identifyTruePositive(file, globalFrameCount);
			
			// Object handover
			objectHandover();

			// 3D estimation
			compute3D_Coords();
				
			// Camera handoff for frame t + 1
			cameraHandoff();

		}

		void objectHandover() {

			// Has the True Positive been located ?
			for (auto cc : currCandidates) if (cc->isRealBall) return;
			if (finalCoords.size() < 2) return;

			// Direction of travel based on last known set of coordinates
			double velX = (finalCoords.end() - 1)->x - (finalCoords.end() - 2)->x;

			// Possible set of states the ball could be in
			if (transitRegion_A.contains(Point(finalPoint3D.x, finalPoint3D.y)) && velX > 0) STATE = 1;
			if (transitRegion_A.contains(Point(finalPoint3D.x, finalPoint3D.y)) && velX < 0) STATE = 3;

			if (transitRegion_B.contains(Point(finalPoint3D.x, finalPoint3D.y)) && velX > 0) STATE = 3;
			if (transitRegion_B.contains(Point(finalPoint3D.x, finalPoint3D.y)) && velX < 0) STATE = 5;

			// loop through all currCandidates.
			for (unsigned i = 0; i < currCandidates.size(); i++)
			{
				if (abs(currCandidates[i]->cameraID - STATE) <= 1) currCandidates[i]->isRealBall = true;
			}
			
		}

		//=========================================================================================
		void finalizeResults(Mat& modelPreview, Mat& cameraView) {

			/********************************************************************************
												3D DRAWING
			*********************************************************************************/

			// Display ball location and trajectory on the field model
			Mat img = fieldModel.clone();
			//for (auto candidate : currCandidates)
			//{
			//	// Indicate position
			//	circle(img, candidate->getLastPoint(), 10, CV_RGB(0, 0, 0), -1);
			//	circle(img, candidate->getLastPoint(), 10, CV_RGB(255, 0, 0), 2);

			//	char ID[40];
			//	sprintf(ID, "%d", candidate->cameraID + 1);
			//	putText(img, ID, candidate->getLastPoint() + Point2f(-40, 10), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 2);

			//	// Display Trajectory
			//	//drawTraceFiltered(img, candidate);
			//}

			// Display player location and trajectory on the field model
			/*for (int i = 0; i < CAMERAS_CNT; i++)
			{
				for (auto candidate : currPlayerCandidates[i])
				{
					drawPoint(img, candidate, 15, -1);
				}
			}*/

			// Display player location
			for (int i = 0; i < updatedPlayerCand.size(); i++)
			{
				drawPoint(img, updatedPlayerCand[i], 15, -1);
			}

			/*for (int i = 0; i < estimatedTrajectory.size(); i++)
			{
				int x = estimatedTrajectory[i].x;
				int y = estimatedTrajectory[i].y;

				circle(img, Point(x, y), 5, Scalar(0, 0, 255));
			}

			//circle(img, landingPos, 30, CV_RGB(0, 0, 0), 1);

			//circle(img, Point(0, 0), 300, (0, 0, 0));*/

			//Display coordinates of true positive on the field model
			for (auto real : getTruePositives())
			{
				if (real->coords3D.size() == 0) break;
				double x = (real->coords3D.end() - 1)->x / 0.05464;
				double y = (real->coords3D.end() - 1)->y / 0.06291;

				circle(img, Point(x, y), 10, CV_RGB(0, 0, 0), -1);
				circle(img, Point(x, y), 10, CV_RGB(255, 0, 0), 2);

				break;
			}

			// For visual debugging
			//for (auto real : getTruePositives())
			//{
			//	if (real->other_coord.first != Point())
			//	{
			//		Point coord = Point(real->other_coord.first) / 3; // convert to pixels
			//		int camID = real->other_coord.second;

			//		for (auto cam : real->cameraVisible) if (cam->id == camID)
			//		{
			//			circle(cameraView, coord + cam->viewRect.tl(), 10, (0, 0, 0), 1);
			//		}
			//	}
			//}

			// Save track Data
			saveTrackData();

			modelPreview = img;

			/*#ifdef DISPLAY_GROUND_TRUTH
			if (GTfinalPoint3D != Point3d())
			{
				char coordinate[40];
				sprintf(coordinate, "%.0f %.0f %.0f", GTfinalPoint3D.x, GTfinalPoint3D.y, GTfinalPoint3D.z);
				putText(img, coordinate, Point((int)GTfinalPoint3D.x + 20, (int)GTfinalPoint3D.y + 40), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
				circle(img, Point(GTfinalPoint3D.x, GTfinalPoint3D.y), 3, Scalar(0, 0, 255));
			}
			#endif*/

			//return img;
		}

		//=========================================================================================

		void saveTrackData() {

			static ofstream outFile;
			static stringstream sstm;
			static bool Switch = false;

			// Initialize file name (one time)
			if (Switch == false)
			{
				sstm.str("");
				sstm << "3D Data " << ".txt";
				outFile.open(sstm.str());

				Switch = true;
			}
			
			outFile << finalPoint3D.x << " " << finalPoint3D.y << " " << finalPoint3D.z << endl;

		}

		//=========================================================================================
		
		void compute3D_Coords() {

			/*#ifdef DISPLAY_GROUND_TRUTH
			if (GroundTruth.size() == 2)	GTfinalPoint3D = Triangulate(GroundTruth[0]->camCoords, Point3d(GroundTruth[0]->GTcoord.x, GroundTruth[0]->GTcoord.y, 0), GroundTruth[1]->camCoords, Point3d(GroundTruth[1]->GTcoord.x, GroundTruth[1]->GTcoord.y, 0));
			else							GTfinalPoint3D = Point3d();
			#endif*/

			/********************************************************************************
									Prepare for height estimation ( frame t )
			*********************************************************************************/

			// Re-initialize temporary container, detection count, 3D coords
			vector<ProjCandidate*> tempBall;
			int detections = 0;
			finalPoint3D = Point3d();

			// Push truePositives into container and increment detection count
			for (auto cc : currCandidates) if (cc->isRealBall) 
			{ 
				detections++; 
				tempBall.push_back(cc); 
			}

			/********************************************************************************
										Triangulate if 2 detections
			*********************************************************************************/
			if (detections == 2)
			{
				// Destroy planeTrajectory
				planeTrajectory = Mat();
				//velocity = Point3d();

				Point2f p1 = *(tempBall[0]->coords_meters.end() - 1);
				Point2f p2 = *(tempBall[1]->coords_meters.end() - 1);

				finalPoint3D = Triangulate(tempBall[0]->camCoords, Point3d(p1.x, p1.y, 0), tempBall[1]->camCoords, Point3d(p2.x, p2.y, 0)).second;

				// Transfer coordinates back to object if valid
				if (finalPoint3D.x > 0 && finalPoint3D.y > 0 && finalPoint3D.x < 105 && finalPoint3D.y < 68)
				{
					for (auto cc : currCandidates) if (cc->isRealBall) cc->coords3D.push_back(finalPoint3D);
					finalCoords.push_back(finalPoint3D);
				}

				return;
			}

			/********************************************************************************
								Internal Height Estimation if 1 detection
			*********************************************************************************/                           
			else if (detections == 1)
			{

				// If plane not formed & At least 2 points are available for the formation of a Plane
				if (planeTrajectory.empty() && tempBall[0]->coords3D.size() > 1)
				{
					// Form plane
					planeTrajectory = formPlane(tempBall[0]->coords3D);

					// Kinematics analysis
					//establishTrajectory(tempBall[0]->coords3D);
				}

				// Estimate 3D coordinates if plane exists
				if (!planeTrajectory.empty())
				{
					// Current coordinates at time t
					Point2f projBallCoords = *(tempBall[0]->coords_meters.end() - 1);

					// Camera coordinates
					Point3d camCoords = tempBall[0]->camCoords;
						
					// Estimate 3D coordinates
					finalPoint3D = InternalHeightEstimation(camCoords, projBallCoords, planeTrajectory);
					
					// Transfer coordinates back to object if valid
					if (finalPoint3D.x > 0 && finalPoint3D.y > 0 && finalPoint3D.x < 105 && finalPoint3D.y < 68)
					{
						for (auto cc : currCandidates) if (cc->isRealBall) cc->coords3D.push_back(finalPoint3D);
						finalCoords.push_back(finalPoint3D);
					}

					// If finalPoint3D != 0, opposite camera will then predict its 2D position in meterts
					for (int i = 0; i < tempBall[0]->cameraVisible.size(); i++)
					{
						if (tempBall[0]->cameraVisible[i]->id != tempBall[0]->cameraID)
						{
							vector<Point2f> p_orig = vector<Point2f>(1);
							vector<Point2f> p_proj = vector<Point2f>(1);

							p_orig[0] = Inv_Triangulate(tempBall[0]->cameraVisible[i]->camCoords, finalPoint3D);
							p_orig[0] = Point(p_orig[0].x / 0.05464, p_orig[0].y / 0.06291);
							
							perspectiveTransform(p_orig, p_proj, tempBall[0]->cameraVisible[i]->homography.inv());

							currCandidates[0]->other_coord.first  = p_proj[0];
							currCandidates[0]->other_coord.second = tempBall[0]->cameraVisible[i]->id;
						}
					}
				}

				// Height = 0 if all fails
				else
				{
					finalPoint3D = Point3d((tempBall[0]->coords_meters.end() - 1)->x, (tempBall[0]->coords_meters.end() - 1)->y, 0);
					
					// Transfer coordinates back to object if valid
					if (finalPoint3D.x > 0 && finalPoint3D.y > 0 && finalPoint3D.x < 105 && finalPoint3D.y < 68)
					{
						for (auto cc : currCandidates) if (cc->isRealBall) cc->coords3D.push_back(finalPoint3D);
						finalCoords.push_back(finalPoint3D);
					}
				}
			}				
	

		}

		//=========================================================================================
		void establishTrajectory(vector<Point3d> coords) {

			// Position of ball in meters
			Point3d currentPos	= *(coords.end() - 1);
			Point3d prevPos		= *(coords.end() - 2);

			// Ball velocity m/f
			velocity = currentPos - prevPos;
			
			// Ball velocity m/s
			velocity = velocity * 25;

			// Landing time in seconds
			landingTime = abs(2 * velocity.z / 9.81);

			landingPos = Point(velocity.x * landingTime + currentPos.x, velocity.y * landingTime + currentPos.y);
			landingPos = Point(landingPos.x * 18.3, landingPos.y * 15.9);

			cout << landingPos << endl;
			
			// Landing time in frames
			int landingTime_f = landingTime * 25;

			// Establish trajectory in pixels
			estimatedTrajectory = vector<Point3d>();

			// Ball velocity p/f
			velocity.x = velocity.x * 18.3 / 25;
			velocity.y = velocity.y * 15.9 / 25;

			cout << velocity.z << endl;

			for (int t = 0; t < landingTime_f; t++)
			{
				float x = velocity.x * t + currentPos.x * 18.3;
				float y = velocity.y * t + currentPos.y * 15.9;
				float z = velocity.z * t - 0.00785 * pow(t,2);

				estimatedTrajectory.push_back(Point3d(x, y, z));
			}
		}

		//=========================================================================================
		inline void drawPolygon (Mat& frame, ProjCandidate* cand) {
			vector<Point2f> points = cand->getLastPolygon().points;
			Scalar color = colors[cand->cameraID];
			if (points.empty()) return;
			for (unsigned i = 1; i < points.size(); i++) {
				line(frame, points[i-1], points[i], color, 1, CV_AA);
			}
			line(frame, points[points.size()-1], points[0], color, 1, CV_AA);
		}

		//=========================================================================================
		inline void drawPoint (Mat& image, ProjCandidate* cand, int rad = 1, int thickness = 1) {

			if (cand->coords.size() < 1 || cand->coords.size() > 2000) return;

			if (thickness == 1) circle(image, *(cand->coords.end()-1), rad, colors[cand->cameraID], 1, CV_AA);
			if (thickness == -1) circle(image, *(cand->coords.end() - 1), rad, teamColors[cand->teamID], thickness, CV_AA);

			/*char ID[40];
			sprintf(ID, "%d", cand->cameraID + 1);
			putText(image, ID, cand->getLastPoint() + Point2f(-10,10), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 2);*/
		}

		//=========================================================================================
		inline void drawBall(Mat& image, ProjCandidate* cand, int rad = 1, int thickness = 1) {

			if (cand->coords.size() < 1 || cand->coords.size() > 2000) return;

			if (thickness == -1) circle(image, *(cand->coords.end() - 1), rad, colors[cand->cameraID], thickness, CV_AA);

			/*char ID[40];
			sprintf(ID, "%d %d", cand->ID, cand->ID2);
			putText(image, ID, cand->getLastPoint(), CV_FONT_HERSHEY_SIMPLEX, 2, CV_RGB(0, 0, 0));*/
		}

		//=========================================================================================
		inline void drawTrace (Mat& image, ProjCandidate* cand) {
			vector<Point2f>& vctr = cand->coords;
			for (unsigned i = vctr.size()-1; i > 0; i--) {
				if (cand->frames[i-1] != cand->frames[i]-1) {
					break;
				} else {
					line(image, vctr[i-1], vctr[i], colors[cand->cameraID], 1, CV_AA);
				}
			}
		}
		
		//=========================================================================================
		inline void drawTraceFiltered (Mat& image, ProjCandidate* cand) {
			vector<Point2f>& vctr = cand->coordsKF;
			//vector<Point2f>& vctr = cand->coords;

			for (unsigned i = vctr.size()-1; i > 0; i--) 
			{
				if (cand->frames[i-1] != cand->frames[i]-1) 
				{
					break;
				} 
				
				else 
				{
					line(image, vctr[i-1], vctr[i], colors[cand->cameraID], 1, CV_AA);
				}
			}
		}

		//=========================================================================================
		void identifyTruePositive(ofstream& file, int globalFrameCount) {
			
			// return if no detections
			if (currCandidates.size() <= 1)
			{
				if (globalFrameCount >= 358) file << 0 << " " << framesProcessed << endl;
				return;
			}

			double minVal = numeric_limits<double>::max();
			ProjCandidate *resCand1 = NULL, *resCand2 = NULL;

			/***************************************************
						Epipolar Geometry for Ball
			****************************************************/
			for (unsigned i = 0; i < currCandidates.size() - 1; i++)
			{
				for (unsigned j = i + 1; j < currCandidates.size(); j++)
				{
					auto cand1 = currCandidates[i];
					auto cand2 = currCandidates[j];

					Point3d cand1Coords = Point3d((cand1->coords_meters.end() - 1)->x, (cand1->coords_meters.end() - 1)->y, 0);
					Point3d cand2Coords = Point3d((cand2->coords_meters.end() - 1)->x, (cand2->coords_meters.end() - 1)->y, 0);

					pair < double, Point3d > comp_LastPoint = Triangulate(cand1->camCoords, cand1Coords, cand2->camCoords, cand2Coords);

					if (comp_LastPoint.first < minVal && comp_LastPoint.second.z > 0)
					{
						minVal = comp_LastPoint.first;
						resCand1 = cand1;
						resCand2 = cand2;
					}
				}
			}
			
			if (minVal < 1)
			{
				if (globalFrameCount >= 358) file << minVal << " " << framesProcessed << endl;
				resCand1->isRealBall = true;
				resCand2->isRealBall = true;
			}

			else if(minVal < 2)
			{
				if (globalFrameCount >= 358) file << minVal << " " << framesProcessed << endl;
			}

			else
			{
				if (globalFrameCount >= 358) file << 0 << " " << framesProcessed << endl;
			}

			// OPTIMIZE THIS SHIT
			vector<ProjCandidate*> allPlayerCandidates;
			allPlayerCandidates.reserve(currPlayerCandidates[0].size() + currPlayerCandidates[1].size() + currPlayerCandidates[2].size() + currPlayerCandidates[3].size() + currPlayerCandidates[4].size() + currPlayerCandidates[5].size()); // preallocate memory
			allPlayerCandidates.insert(allPlayerCandidates.end(), currPlayerCandidates[0].begin(), currPlayerCandidates[0].end());
			allPlayerCandidates.insert(allPlayerCandidates.end(), currPlayerCandidates[1].begin(), currPlayerCandidates[1].end());
			allPlayerCandidates.insert(allPlayerCandidates.end(), currPlayerCandidates[2].begin(), currPlayerCandidates[2].end());
			allPlayerCandidates.insert(allPlayerCandidates.end(), currPlayerCandidates[3].begin(), currPlayerCandidates[3].end());
			allPlayerCandidates.insert(allPlayerCandidates.end(), currPlayerCandidates[4].begin(), currPlayerCandidates[4].end());
			allPlayerCandidates.insert(allPlayerCandidates.end(), currPlayerCandidates[5].begin(), currPlayerCandidates[5].end());

			/***************************************************
						Epipolar Geometry for Player
			****************************************************/

			//const clock_t begin_time = clock();

			ProjCandidate* cand1 = NULL;
			ProjCandidate* cand2 = NULL;
			
			updatedPlayerCand = vector<ProjCandidate*>();
			vector<int> used;
			unsigned i, j;

			for (i = 0; i < allPlayerCandidates.size() - 1; i++)
			{
				// If candidate already used
				if (find(used.begin(), used.end(), i) != used.end()) continue;

				// Else initialize
				cand1 = allPlayerCandidates[i];
				vector<double> distance;

				Point2f cand1Coords, cand2Coords;

				int offset = i + 1;
				// Loop through remaining players
				for (j = i + 1; j < allPlayerCandidates.size(); j++)
				{
					// If candidate already used
					if (find(used.begin(), used.end(), j) != used.end())
					{
						distance.push_back(100);
						continue;
					}

					else
					{
						// Compute distance
						cand2 = allPlayerCandidates[j];

						cand1Coords = Point2f((cand1->coords_meters.end() - 1)->x, (cand1->coords_meters.end() - 1)->y);
						cand2Coords = Point2f((cand2->coords_meters.end() - 1)->x, (cand2->coords_meters.end() - 1)->y);

						distance.push_back(norm(cand1Coords - cand2Coords));
					}
				}

				// If last player
				if (distance.size() == 0)
				{
					updatedPlayerCand.push_back(cand1);
					used.push_back(i);
				}
				
				// Get nearest player
				else
				{
					float	min_distance = *min_element(distance.begin(), distance.end());
					int		min_index = min_element(distance.begin(), distance.end()) - distance.begin();
					
					// Update player pair
					if (min_distance < 5)
					{
						cand1->ID = min(cand1->ID, cand2->ID);
						cand1->ID2 = max(cand1->ID, cand2->ID);
						updatedPlayerCand.push_back(cand1);

						used.push_back(i); 
						used.push_back(min_index + offset);
					}

					else
					{
						updatedPlayerCand.push_back(cand1);
						used.push_back(i);
					}
				}

				//cout << (*(updatedPlayerCand.end() - 1))->teamID;
			}

			//cout << "pc size " << allPlayerCandidates.size() << " used size " << used.size() << " upc size " << updatedPlayerCand.size() << endl;
		}

		//=========================================================================================
		void cameraHandoff() {

			Rect boundary = Rect(0, 0, 960, 540);

				// Initialize temporary container
				vector<int> cameraID;
				vector<Camera*> tempCameraId;
								
				// Loop through current candidates
				for (int i = 0; i < currCandidates.size(); i++)
				{
					// if current candidate contains the true positive
					if (currCandidates[i]->isRealBall)
					{
						// its current projected and prediction coordinates
						vector<Point2f> curCrd(1, currCandidates[i]->coords.back());
						vector<Point2f> nxtCrd(1, currCandidates[i]->coords_pred);

						vector<Point2f> _curCrd(1);
						vector<Point2f> _nxtCrd(1);

						// Loop through all cameras
						for (int j = 0; j < CAMERAS_CNT; j++)
						{
							// Re project true positive back into all frames ( Inv Homography )
							perspectiveTransform(curCrd, _curCrd, cameras[j]->homography.inv());
							perspectiveTransform(nxtCrd, _nxtCrd, cameras[j]->homography.inv());

							// Down scale
							_curCrd[0] = _curCrd[0] / 2;
							_nxtCrd[0] = _nxtCrd[0] / 2;

							// Store camera ID if 
							if (currCandidates[i]->coords3D.size() != 0)
							// A) ball near/within field of view  B) coordinates of ball not (0,0,0)  C) camera ID not yet stored
							if ((boundary.contains(_curCrd[0]) || boundary.contains(_nxtCrd[0])) && *(currCandidates[i]->coords3D.end() - 1) != Point3d() && !(std::find(cameraID.begin(), cameraID.end(), j) != cameraID.end())) {
								cameraID.push_back(j);
								tempCameraId.push_back(cameras[j]);
							}
							
						}
					}
				}
				
				for (auto cc : currCandidates)	if (cc->isRealBall)	cc->cameraVisible = tempCameraId;
		}

		//=========================================================================================
		double compareTrajectories_LastPoint (vector<Point2f> traj1, vector<Point2f> traj2) {
			Point2f lastCrd1 = *(traj1.end()-1);
			Point2f lastCrd2 = *(traj2.end()-1);
		
			double distSQ = (lastCrd1.x - lastCrd2.x) * (lastCrd1.x - lastCrd2.x) + (lastCrd1.y - lastCrd2.y) * (lastCrd1.y - lastCrd2.y);
			double dist = sqrt(distSQ);
			return dist;
		}

		//=========================================================================================
		Point2f getCenterOfMass (vector<Point2f> traj, int lh) {
			// ----- calculates the center of mass for given trajectory for the last 'lh' points 
			
			if (traj.empty()) {
				return Point2f();
			} else if (traj.size() == 1) {
				return traj[0];
			}

			int startIdx = int(std::max(double(traj.size())-lh, 0.0));

			double s1_x = 0, s1_y = 0, s2_x = 0, s2_y = 0;
			for (int i = startIdx; i < int(traj.size())-1; i++) {
				s1_x += (traj[i+1].x * traj[i+1].x  - traj[i].x * traj[i].x);
				s1_y += (traj[i+1].y * traj[i+1].y  - traj[i].y * traj[i].y);
				s2_x += (traj[i+1].x - traj[i].x);
				s2_y += (traj[i+1].y - traj[i].y);
			}
			double ctr_x, ctr_y;
			if (s2_x < 0.05) {
				ctr_x = (traj[startIdx].x + traj[traj.size()-1].x) / 2;
			} else {
				ctr_x = s1_x / (2 * s2_x);
			}
			if (s2_y < 0.05) {
				ctr_y = (traj[startIdx].y + traj[traj.size()-1].y) / 2;
			} else {
				ctr_y = s1_y / (2 * s2_y);
			}
			return Point2f(float(ctr_x), float(ctr_y));
		}

		//=========================================================================================
		pair < double, Point3d > Triangulate(Point3d camTop, Point3d ballTop, Point3d camBtm, Point3d ballBtm) {

			Point3d P1 = camTop; // Point P1
			Point3d P2 = camBtm; // Point P2

			Point3d L1 = ballTop - P1; // Vector L1
			Point3d L2 = ballBtm - P2; // Vector L2

			// Perpendicular vector from L1 to L2 [ PQ = L2.xyz - L1.xyz ]
			Mat PQ = (Mat_<float>(3, 3) <<

				// Constant ---- // L2 variable (s) --- // L1 variable (-t)
				   P2.x - P1.x,		L2.x,			      -L1.x,
				   P2.y - P1.y,		L2.y,			      -L1.y,
				   P2.z - P1.z,		L2.z,			      -L1.z);

			/********************************************************************************
									topRow = PQ variables (dot) Vector L1
									btmRow = PQ variables (dot) Vector L2 

									P      = PQ constants (dot) Vector L1 L2

										| topRow | * | s | =  P
										| btmRow |   | t |   

			*********************************************************************************/

			Mat topRow = (Mat_<float>(1, 2) <<

				PQ.at<float>(0, 1) * L1.x + PQ.at<float>(1, 1) * L1.y + PQ.at<float>(2, 1) * L1.z,
				PQ.at<float>(0, 2) * L1.x + PQ.at<float>(1, 2) * L1.y + PQ.at<float>(2, 2) * L1.z);

			Mat btmRow = (Mat_<float>(1, 2) <<

				PQ.at<float>(0, 1) * L2.x + PQ.at<float>(1, 1) * L2.y + PQ.at<float>(2, 1) * L2.z,
				PQ.at<float>(0, 2) * L2.x + PQ.at<float>(1, 2) * L2.y + PQ.at<float>(2, 2) * L2.z);

			Mat B = (Mat_<float>(2, 1) <<

				-1 * (PQ.at<float>(0, 0) * L1.x + PQ.at<float>(1, 0) * L1.y + PQ.at<float>(2, 0) * L1.z),
				-1 * (PQ.at<float>(0, 0) * L2.x + PQ.at<float>(1, 0) * L2.y + PQ.at<float>(2, 0) * L2.z));

			/********************************************************************************

										| topRow | * | s | =  P
										| btmRow |   | t |

										                DQ = P
											             Q = D.inv() * P , where Q = [ s, t ]

														Ax = B
														 x = A.inv() * B , where x = [ s, t ]

			*********************************************************************************/

			Mat A, x; 
			vconcat(topRow, btmRow, A); 
			x = A.inv() * B; 	

			// L1_t = L1(t) + P1
			Mat L1_t = (Mat_<float>(1, 3) << P1.x + L1.x * x.at<float>(1), P1.y + L1.y * x.at<float>(1), P1.z + L1.z * x.at<float>(1));
			Mat L2_s = (Mat_<float>(1, 3) << P2.x + L2.x * x.at<float>(0), P2.y + L2.y * x.at<float>(0), P2.z + L2.z * x.at<float>(0));
	
			// Compute perpendicular distance of line PQ and the mid point of it
			double euclideanDist = norm(L1_t, L2_s, NORM_L2);
			Point3d midPoint = Point3d((L1_t.at<float>(0, 0) + L2_s.at<float>(0, 0)) / 2, (L1_t.at<float>(0, 1) + L2_s.at<float>(0, 1)) / 2, (L1_t.at<float>(0, 2) + L2_s.at<float>(0, 2)) / 2);
		
			pair < double, Point3d > result = make_pair(euclideanDist, midPoint);
			
			return result;			
		}

		//=========================================================================================
		vector<ProjCandidate*> getTruePositives(){

			vector<ProjCandidate*> Ball;

			for (auto cc : currCandidates)	if (cc->isRealBall)	Ball.push_back(cc);

			return Ball;

			//// Update Ball if new ID
			//for (int i = 0; i < currCandidates.size(); i++)
			//{
			//	if (currCandidates[i]->isRealBall)
			//	{
			//		Ball.push_back(currCandidates[i]);
			//	}
			//}
		}

		//=========================================================================================
		Point Inv_Triangulate(Point3d cam, Point3d ball) {

			// Vector of line from camera to ball
			Point3d vector = ball - cam;

			// Parametric Equation of the Line
			Mat P_Line = (Mat_<float>(3, 2) <<

				cam.x, vector.x,
				cam.y, vector.y,
				cam.z, vector.z);

			// Value of t at which Z component = 0
			float t = cam.z / -vector.z;

			// Coordinate at which the line intersects the XY plane
			float x = cam.x + vector.x * t;
			float y = cam.y + vector.y * t;

			return Point(x, y);
		}

		//=========================================================================================
		Mat formPlane(vector<Point3d> points) {
			
			Point3d A(*(points.end() - 2));
			Point3d B(*(points.end() - 1));
			
			cv::Mat temp(3, 1, CV_64FC1);
			const double *data = temp.ptr<double>(0);

			// Vector on the plane... Task -> Least Squares Fit / RANSAC ?
			Point3d AB = B - A;
			
			// Normal vector of AB lying on the ground plane n = (a, b, 0)
			Point3d N(-AB.y, AB.x, 0);

			// Equation of the plane ax + by + cz - constant = 0... Task -> Mean / subset of RANSAC ?
			Mat Plane = (Mat_<float>(1, 4) << N.x, N.y, 0, -(N.x * A.x + N.y * A.y));

			return Plane;
		}

		//=========================================================================================
		Point3d InternalHeightEstimation(Point3d cam, Point2f ball, Mat _plane) {

			Point3d C = cam;
			Point3d A(ball.x, ball.y, 0);

			// Projection of cam on the XY plane, Point D
			Point3d D(C.x, C.y, 0);

			// Vector from projected camera to ball, Vector DA
			Point3d DA = A - C;

			// Parametric equation of line DA
			// xyz = aT + constant
			Mat lineDA = (Mat_<float>(3, 2) << DA.x, C.x,
											   DA.y, C.y,
										       DA.z, C.z);

			double t = (-_plane.at<float>(0, 3) - C.x*_plane.at<float>(0, 0) - C.y*_plane.at<float>(0, 1)) / (DA.x*_plane.at<float>(0, 0) + DA.y*_plane.at<float>(0, 1));

			// Point of intersection, E
			Point3d E(DA.x * t + C.x, DA.y * t + C.y, DA.z * t + C.z);

			return E;
		}

		//=========================================================================================
		~MultiCameraTracker(void) {
			for (auto i : currCandidates) {
				delete i;
			}
		}

};

}