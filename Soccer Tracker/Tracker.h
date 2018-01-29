#pragma once

#include <opencv/cv.h>
#include <utility>
#include <set>

#include "AppearanceAnalyzer.h"
#include "AccuracyMetric.h"
#include "BallCandidate.h"
#include "PlayerCandidate.h"
#include "TemplateGenerator.h"
#include "TrackInfo.h"
#include "MultiCameraTracker.h"
#include <fstream>

#include "globalSettings.h"
#include "TrajectoryAnalyzer.h"

using namespace cv;

namespace st {

//*************************************************************************************************
// ----- Is one of the most important classes, performs tracking of ball and players
//*************************************************************************************************
class Tracker {

	//________________________________________________________________________________________________
	private:

		int curFrame, startTracking;
		vector<Mat> ballTempls;
		vector<BallCandidate*> bCandidates;
		vector<ProjCandidate*> Ball;
		vector<vector<BallCandidate*>> bCandidatesGroups;
		BallCandidate* mainCandidate;
		vector<PlayerCandidate*> pCandidates;
		AppearanceAnalyzer appearAnalyzer;
		bool flg;
		Point defRad, iniRad, attachRad, searchIncRad;
		Mat restrictedArea, trajFrame;
		int trajLastFrame;
		Point trajLastPoint;
		double M1_loose_threshold, M1_find_threshold, perspectiveRatio;
		TrackInfo trackInfo;
		TRACKER_STATE trackerState;
		vector<Point> mainCandidateTraj, givenTrajectory;
		AccuracyMetric metric;

		// Cooperative Tracking
		vector<Rect> region;
		int Location = 0;

		// Kick off
		int count = 0;
		Point lastBallLoc;
	//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		Tracker() {}

		//=========================================================================================
		void ball_addCandidateManually (int x, int y) 
		{
			BallCandidate* bc = new BallCandidate(curFrame, Point(x,y), defRad, 0.0);
			bc->switchState(curFrame, BALL_STATE::TRACKING);
			bCandidates.push_back(bc);
			mainCandidate = NULL;
		}

		//=========================================================================================
		void initialize (int TID) {

			appearAnalyzer.setBallTempls(ballTempls);
			appearAnalyzer.generateClassifier();

			curFrame = 0;
			flg = true;

			defRad       = Point(45, 35);  // Initial search radius (45, 35)
			iniRad       = Point(22, 22);  //  
			attachRad    = Point(100, 60);
			searchIncRad = Point(15,15);   // Rate at which size of search box increases

			M1_loose_threshold = 0.94;
			M1_find_threshold = 0.96;
			
			// Set restricted zone
			restrictedArea = Mat(fSize, CV_32FC1, 1.0);

			restrictedArea(Rect(0, 0, fSize.x, int(0.08*fSize.y))) = 0.0;
			//restrictedArea(Rect(0, int(0.93*fSize.y), fSize.x, int(0.07*fSize.y))) = 0.0;

			if (TID == 2)	{restrictedArea(Rect(Point(435, 510), Point(550, 540))) = 0.0;}
			if (TID == 3)   {restrictedArea(Rect(Point(190, 525), Point(215, 540))) = 0.0; restrictedArea(Rect(Point(590, 510), Point(610, 525))) = 0.0; }

			appearAnalyzer.setRestrictedArea(restrictedArea);

			// Test
			vector<Rect> _region;
			_region.push_back(Rect(Point(60, 0), Point(105, 68)));
			_region.push_back(Rect(Point(60, 0), Point(105, 68)));
			_region.push_back(Rect(Point(28, 0), Point(76, 68)));
			_region.push_back(Rect(Point(28, 0), Point(76, 68)));
			_region.push_back(Rect(Point(0, 0) , Point(50, 68)));
			_region.push_back(Rect(Point(0, 0) , Point(50, 68)));
			region = _region;

			trajFrame = Mat(fSize.y, fSize.x, CV_8UC3, CV_RGB(255,255,255));

			mainCandidate = NULL;
			trackerState = TRACKER_STATE::BALL_NOT_FOUND;
		}

		//=========================================================================================
		void setBallTempls (vector<Mat>& ballTempls) {
			this->ballTempls = ballTempls;
			appearAnalyzer.setBallTempls(ballTempls);
		}

		//=========================================================================================
		void setGivenTrajectory (vector<Point>& trajectory) {
			givenTrajectory = trajectory;
			TrajectoryAnalyzer::drawTrajectory(trajFrame, trajectory, 2, CV_RGB(255,0,0));
		}

		//=========================================================================================
		void setPerspectiveRatio (double perspectiveRatio) {
			this->perspectiveRatio = perspectiveRatio;
		}

		//=========================================================================================
		void setTrackInfo (TrackInfo tInfo) {
			
		}

		//=========================================================================================
		TrackInfo getTrackInfo () {


			if (mainCandidate == NULL || mainCandidate->getState() == 1) 
			{
				trackInfo.set(-1, Rect(), Point(-1, -1), Point(), givenTrajectory[curFrame + 2], pCandidates);
			} 
			
			else 
			{
				trackInfo.set(mainCandidate->id, mainCandidate->curRect, mainCandidate->curCrd, mainCandidate->predCrd, givenTrajectory[curFrame + 2], pCandidates);
			}

			return trackInfo;
		}

		//=========================================================================================
		void getTruePositivesTraj (vector<Point>& vctr) {
			vctr = mainCandidateTraj;
		}

		//=========================================================================================
		void getTrajFrame (Mat& fr) {
			fr = trajFrame;
		}

		//=========================================================================================
		void drawTrajectory (Mat& frame, int rad = 1) {

			if (mainCandidate == NULL) 
			{
				return;
			}

			Point curPoint = mainCandidate->curCrd;
			Point prevPoint = trajLastPoint;

			if (curFrame != trajLastFrame + 1) 
			{
				prevPoint = curPoint;
			}

			circle(trajFrame, curPoint, rad, cv::Scalar(255,0,0));
			line(trajFrame, prevPoint, curPoint, cv::Scalar(255, 0, 255));

			trajLastPoint = curPoint;
			trajLastFrame = curFrame;
		}

		//=========================================================================================
		inline void drawBallGroundTruth (Mat& frame, int rad) {
			Point p = givenTrajectory[curFrame + 3];

			if (p == outTrajPoint) 
			{
				return;
			}

			circle(frame, p, rad, CV_RGB(255,0,0), 1); // Red for ground Truth
		}

		//=========================================================================================
		
		inline void drawDistance (Mat& frame) {
			Point p = givenTrajectory[curFrame + 3];

			if ((mainCandidate != NULL) && (p != outTrajPoint)) 
			{
				circle(frame, mainCandidate->curCrd, 3, CV_RGB(0, 255, 0), 3);
				line(frame, mainCandidate->curCrd, p, CV_RGB(255,0,255), 2);
			}
		}

		//=========================================================================================
		int inline distSQ (Point& p1, Point& p2) {
			return int((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
		}

		//=========================================================================================
		inline void updateMetric(ofstream& file, int processedFrames) {

			if (processedFrames < 358) return; // 358

			Point p = givenTrajectory[curFrame + 3];
			Point mainCandCrd;

			bool exists = (p != outTrajPoint);
			bool detected = (mainCandidate != NULL);				

			// Temporary container
			if (detected) mainCandCrd = mainCandidate->curCrd;
			else		  mainCandCrd = Point(-1, -1);

			/************************************************
							Ground Truth exists
			*************************************************/
			if (exists)
			{	
				metric.TG++;

				/************************************************
								Candidate Found
				*************************************************/
				if (detected) 
				{
					// Compute euclidean distance
					double dist = sqrt(distSQ(mainCandidate->curCrd, p));

					// Update track ratio of main candidate
					if (mainCandidate->getState() == BALL_STATE::TRACKING) { metric.addDistance(dist); metric.TR = double(mainCandidate->getStateDuration(BALL_STATE::TRACKING)) / mainCandidate->lifeTime; }
					else												     metric.addDistance(0.0);
					
					if (dist < 10) 	metric.TP++; // correctly detected
					else       	    metric.FN++; // incorrectly detected

					// Save results to file
					//file << p.x << " " << p.y << " " << mainCandidate->curCrd.x << " " << mainCandidate->curCrd.y << endl;
				} 
				
				/************************************************
								Candidate Not Found
				*************************************************/
				else 
				{
					// Save results to file
					//file << p.x << " " << p.y << " " << -1 << " " << -1 << endl;

					metric.addDistance(0.0);
					metric.FN++;
				}

			} 
			
			/************************************************
						Ground Truth does not exist
			*************************************************/
			else 
			{
				// Save results to file
				//file << p.x << " " << p.y << " " << -1 << " " << -1 << endl;

				metric.addDistance(0.0);
				
				if (detected)	metric.FP++; // incorrectly detected
				else 			metric.TN++; // no candidate found
			}

			/************************************************
							Update Metrics
			*************************************************/
			
			double TP = metric.TP;
			double TN = metric.TN;
			double FP = metric.FP;


			// Tracker Detection Rate
			if (metric.TG != 0)				metric.Tracker_Detection_Rate   = metric.TP / (double)metric.TG;

			// False Alarm Rate
			if (metric.TP + metric.FP != 0)	metric.FA_Rate					= metric.FP / (double)(metric.TP + metric.FP);

			// Precision and Recall
			if (metric.TP + metric.FN != 0)	metric.Recall					= metric.TP / (double)(metric.TP + metric.FN);
			if (metric.TP + metric.FP != 0)	metric.Positive_Precision		= metric.TP / (double)(metric.TP + metric.FP);
			if (metric.FN + metric.TN != 0)	metric.Negative_Precision		= metric.TN / (double)(metric.FN + metric.TN);

			// TN FN FP Rate
			if (metric.FP + metric.TN != 0)	metric.TN_Rate					= metric.TN / (double)(metric.FP + metric.TN);
			if (metric.FN + metric.TP != 0)	metric.FN_Rate					= metric.FN / (double)(metric.FN + metric.TP);
			if (metric.FP + metric.TN != 0)	metric.FP_Rate					= metric.FP / (double)(metric.FP + metric.TN);

			// Accuracy (Initialize only after TG > 0)
			//if (metric.TG != 0)				metric.Accuracy = (metric.TP + metric.TN) / curFrame; // cannot curFrame

			file << p.x << " " << p.y << " " << mainCandCrd.x << " " << mainCandCrd.y << " "
				<< metric.Tracker_Detection_Rate << " " << metric.FA_Rate << " " << metric.Recall << " "
				<< metric.Positive_Precision << " " << metric.Negative_Precision << " "
				<< metric.TN_Rate << " " << metric.FN_Rate << " " << metric.FP_Rate << " "
				<< processedFrames << endl;

			//file << p.x << " " << p.y << metric.Recall << metric.FN_Rate << endl;
		}

		//=========================================================================================
		void processFrame(Mat& frame, vector<Point>& ball_cand, vector<Rect>& player_cand, vector<ProjCandidate*> Ball, int TID, int processedFrames, ofstream& file, Mat mask) {

			this->Ball = Ball;
			trackPlayers(player_cand, frame, TID, mask);
			trackBall(frame, ball_cand, Ball, TID, processedFrames);
			drawTrajectory(frame, 2);
			updateMetric(file, processedFrames);

			#ifdef DISPLAY_GROUND_TRUTH
			drawBallGroundTruth(frame, 5);
			//drawDistance(frame);
			#endif

			curFrame++;
		}

		//=========================================================================================
		static bool compareMerge_Player (PlayerCandidate* c1, PlayerCandidate* c2) {
			
			// --- check if two players need to be merged
			return (c1->curCrd == c2->curCrd/* && !c1->Occlusion && !c2->Occlusion*/);
		}
		
		//=========================================================================================
		static bool compareMerge_Window (BallCandidate* c1, BallCandidate* c2) {
			
			// --- check if two ball candidates need to be merged
			Rect rectInersection = c1->curRect & c2->curRect;

			if (rectInersection.area() > 0.7*min(c1->curRect.area(), c2->curRect.area()))
			{
				return true; // merge needed
			}

			return false;
		}
		//=========================================================================================
		bool searchSidelines(Mat& frame, int TID, int count, vector<Point>& newCandidates = vector<Point>()) {

			/*****************************************************
			
						Searching for ball at sideline

			******************************************************/
			if (count == 2)
			{
				/**************************************
							Search Players
				***************************************/
				for (auto pc : pCandidates)
				{
					// Player position ( feet )
					Point pPos = pc->curRect.br();

					// If player position is out of bounds (50 and 70)
					if (pPos.y > 515 && pc->curRect.area() > 900 && pc->curRect.width < 50 && pc->curRect.height < 80)
					{
						// Initialize Ball Candidate at head
						pPos = Point(pc->curRect.tl().x + pc->curRect.width / 2, pc->curRect.tl().y);
						BallCandidate* bc = new BallCandidate(curFrame, pPos, iniRad, 0.0);

						// Update search window size
						Rect pRect(pc->curRect);
						#ifdef WINDOW_PERSPECTIVE
						bc->curRad = Point(pRect.width / 2, pRect.height / 2) + perspectiveRad(Point(50, 45), bc->curCrd);
						#else
						bc->curRad = Point(pRect.width / 2, pRect.height / 2) + attachRad;
						#endif // WINDOW_PERSPECTIVE

						bc->fitFrame();

						// Correlate
						vector<Point> matchPoints;
						vector<double> matchProbs;
						appearAnalyzer.getMatches(bc, 1, matchPoints, matchProbs, TID);

						Point nCrd = matchPoints[0];
						double nProb = matchProbs[0];

						// Ball found
						if (nProb > 0.955)
						{
							bc->curCrd = nCrd;
							bc->curAppearM = nProb;
							bc->switchState(curFrame, BALL_STATE::TRACKING);

							// Push back curState many times
							bc->updateStep(30);
							bCandidates.push_back(bc);

							// Update candidate trajectory
							mainCandidateTraj.push_back(bc->curCrd);

							return true;
						}
					}
				}
				
				/**************************************
						Search remaining areas
				***************************************/
				if (newCandidates.size() != 0)
				{
					for (int i = 0; i < newCandidates.size(); i++)
					{
						BallCandidate* bc = new BallCandidate(curFrame, newCandidates[i], iniRad, 0.0);

						// Correlate
						vector<Point> matchPoints;
						vector<double> matchProbs;
						appearAnalyzer.getMatches(bc, 1, matchPoints, matchProbs, TID);

						Point nCrd = matchPoints[0];
						double nProb = matchProbs[0];

						// Ball found
						if (nProb > M1_find_threshold)
						{
							bc->curCrd = nCrd;
							bc->curAppearM = nProb;
							bc->switchState(curFrame, BALL_STATE::TRACKING);

							bc->updateStep();
							bCandidates.push_back(bc);

							// Update candidate trajectory
							mainCandidateTraj.push_back(bc->curCrd);
							return true;
						}
					}

					// no candidate found
					return false;
				}
			}

			/*****************************************************

							Ball found at sideline

			******************************************************/

			if (count == 1)
			{
				BallCandidate* bc = bCandidates[0];

				/*****************************************************
							If player has the ball
				******************************************************/
				for (auto pc : pCandidates)
				{		
					// Iterate through players whose position is out of bounds
					Rect window(pc->curRect.x - pc->curRect.width / 2, pc->curRect.y - pc->curRect.height / 2, pc->curRect.width * 4, pc->curRect.height * 4);
					if (window.contains(bc->curCrd) && pc->curRect.br().y > 510)
					{
						// Update window search size
						#ifdef WINDOW_PERSPECTIVE
						bc->curRad = Point(pc->curRect.width / 2, pc->curRect.height / 2) + perspectiveRad(Point(50, 45), bc->curCrd);
						#else
						bc->curRad = Point(pRect.width / 2, pRect.height / 2) + attachRad;
						#endif // WINDOW_PERSPECTIVE

						bc->fitFrame();

						// Correlate
						vector<Point> matchPoints;
						vector<double> matchProbs;
						appearAnalyzer.getMatches(bc, 1, matchPoints, matchProbs, TID);

						Point nCrd = matchPoints[0];
						double nProb = matchProbs[0];

						// Ball found
						if (nProb > M1_loose_threshold)
						{
							bc->curCrd = nCrd;
							bc->curAppearM = nProb;
							bc->switchState(curFrame, BALL_STATE::TRACKING);
							bc->updateStep();

							// Update candidate trajectory
							mainCandidateTraj.push_back(bc->curCrd);

							// Ball in play, deactivate counter
							if (bc->curCrd.y < 440)
							{
								// set window size to default
								#ifdef WINDOW_PERSPECTIVE
								bc->curRad = perspectiveRad(defRad, bc->curCrd);
								#else
								bc->curRad = defRad;
								#endif // WINDOW_PERSPECTIVE
								return true; //this->count--;
							}
						}

						// Ball not found, assume at head
						else
						{
							Point pHead(pc->curRect.x + pc->curRect.width / 2, pc->curRect.tl().y);
							bc->curCrd = pHead;
							bc->curAppearM = nProb;
							bc->switchState(curFrame, BALL_STATE::TRACKING);
							bc->updateStep();

							mainCandidateTraj.push_back(bc->curCrd);

							return false;
						}
					}
				}

				/*****************************************************
					Ball is at sideline but no player holding onto it
				******************************************************/
				// Correlate
				vector<Point> matchPoints;
				vector<double> matchProbs;
				appearAnalyzer.getMatches(bc, 1, matchPoints, matchProbs, TID);

				Point nCrd = matchPoints[0];
				double nProb = matchProbs[0];

				// Ball found
				if (nProb > M1_find_threshold)
				{
					bc->curCrd = nCrd;
					bc->curAppearM = nProb;
					bc->switchState(curFrame, BALL_STATE::TRACKING);
					bc->updateStep();

					// Update candidate trajectory
					mainCandidateTraj.push_back(bc->curCrd);

					// Ball in play, deactivate counter
					if (bc->curCrd.y < 440) return true;
				}

				// Free ball out of frame
				if ((mainCandidateTraj.end() - 1)->y > 530)
				{
					bc->switchState(curFrame, BALL_STATE::GOT_LOST);
					ball_removeLostCandidates();
					this->count = 2;
					return false;
				}

				return false;
			}
		}


		//=========================================================================================
		void trackBall(Mat& frame, vector<Point>& newCandidates, vector<ProjCandidate*> truePositives, int TID, int processedFrames) {
	
			appearAnalyzer.setFrame(frame); 
			bool Tracking = 0;

			// Get last location of ball
			for (auto b : Ball)
			{
				if (b->coords3D.size() != 0)
				{
					// Ball is being tracked
					Tracking = 1;
					lastBallLoc = Point((b->coords3D.end() - 1)->x, (b->coords3D.end() - 1)->y);
				}
			}

			if (Ball.size() != 0)
			{
				if (Ball[0]->coords3D.size() != 0 && !region[TID].contains(lastBallLoc)) return;
			}

			//if (region[TID].contains(lastBallLoc)) Location = TID;
			//if (Ball.size() == 0) Location = 0;
			//if (!region[TID].contains(lastBallLoc) && Location != 0) return;

			switch (trackerState) {

				//_____________________________________________________________
				case TRACKER_STATE::BALL_NOT_FOUND :

					/*******************************************************************

												Search at side-line

					********************************************************************/
					
					if (!mainCandidateTraj.empty())
					{
						// Last location of ball
						Point currentBallLocation = *(mainCandidateTraj.end() - 1);

						// Ball out of play - Activate side-line search /*currentBallLocation.y > 525*/
						if ((TID == 3 && lastBallLoc.x > 35 && lastBallLoc.x < 45 && lastBallLoc.y < 3 && count == 0) || (TID == 0 && lastBallLoc.x > 90 && lastBallLoc.x < 98 && lastBallLoc.y > 66 && count == 0))
						{
							count = 2; // Active count for current thread
							return;
						}

						// Extract candidates that are out of bounds
						vector<Point> outOfBoundsCandidates;
						for (int i = 0; i < newCandidates.size(); i++)
						{
							if (newCandidates[i].y > 500) 
							{
								outOfBoundsCandidates.push_back(newCandidates[i]);
							}
						}

						// Counter at 2, Search for ball at sideline
						if (count == 2) 
						{
							// Ball found at sidelines
							if (searchSidelines(frame, TID, count, outOfBoundsCandidates))
							{ 
								count--; 
								ball_chooseMainCandidate();
								trackerState = TRACKER_STATE::BALL_FOUND;
								mainCandidateTraj.push_back(mainCandidate->curCrd);
								return;
							}

							// Ball not found
							else return;
						}

					}

					/*******************************************************************
					
												Default Search						
					
					********************************************************************/

					// Destroy ball if it has been tracked over 5 frames but not established as a mainCandidate
					ball_removeOutdatedCandidates(5);

					// Adds Candidates into bCandidates with BALL_STATE::TRACKING
					if (bCandidates.size() < 2) 
					{
						ball_addMoreCandidates(newCandidates, frame, 2, TID);
					}
					
					for (auto bc : bCandidates) 
					{
						ball_updateBallCandidate(bc, frame, TID);
					}
										
					ball_removeLostCandidates();
					ball_removeStuckedCandidates(0.4);
					ball_chooseMainCandidate();

					if (mainCandidate != NULL) 
					{
						trackerState = TRACKER_STATE::BALL_FOUND;
						mainCandidateTraj.push_back(mainCandidate->curCrd);
					}

					break;
				//_____________________________________________________________
				case TRACKER_STATE::BALL_FOUND : 

					// Count at 1, track ball at sideline
					if (count == 1) 
					{						
						if (searchSidelines(frame, TID, count)) count--;
						return;
					}

					for (auto bc : bCandidates) 
					{
						ball_updateBallCandidate(bc, frame, TID);
					}

					ball_removeLostCandidates();
					ball_removeStuckedCandidates(0.4);

					if (mainCandidate == NULL) 
					{
						trackerState = TRACKER_STATE::BALL_NOT_FOUND;
					}

					else
					{
						mainCandidateTraj.push_back(mainCandidate->curCrd);
					}

					break;
			}

			/*if (TID == 3)
				for (int i = 0; i < newCandidates.size(); i++)
					circle(frame, newCandidates[i], 20, CV_RGB(255, 0, 255));*/
			
			/*if (mainCandidate != NULL) 
			{
				mainCandidateTraj.push_back(mainCandidate->curCrd);
			} 

			else 
			{
				mainCandidateTraj.push_back(outTrajPoint);
			}*/
		}

		//=========================================================================================
		inline void updateAttachedHeight (Rect& pRect, BallCandidate* bc, int default = 0) {
			double h;

			int b_y = bc->curCrd.y, p_y_top = pRect.y;

			if (b_y < p_y_top) 
			{
				// --- ball is above the player, attach to the heighest point
				h = 0.0;
			} 
			
			else if (b_y > pRect.y + pRect.height) 
			{
				// --- ball is below the player, attach to the lowest point
				h = 1.0;
			} 

			else 
			{
				// --- ball is within a player, attach to the hit heght
				h = double(b_y - p_y_top) / pRect.height;
			}

			bc->attachedHeight = h;
		}

		//=========================================================================================
		inline Point perspectiveRad (Point curRad, Point crd) {

			double d = double(crd.y) / fSize.y;
			double scale = perspectiveRatio + (1 - perspectiveRatio) * d;
			return Point(int(scale * curRad.x), int(scale * curRad.y));

		}

		//=========================================================================================
		void ball_updateBallCandidate(BallCandidate* bc, Mat& frame = Mat(), int TID = 0, int count = 0) {

			switch (bc->getState()) {

				//_____________________________________________________________
				case BALL_STATE::TRACKING : {
				//*************************************************************

					vector<double> nearbyP_Dist;
					vector<int> nearbyP_Idx;

					/**********************************************************
										Find Nearest Player
					***********************************************************/
					getNearestPlayersVctr(bc, nearbyP_Idx, nearbyP_Dist, 15.0);

					// Locate nearest player
					if (!nearbyP_Dist.empty()) 
					{
						auto nearest     = min_element(nearbyP_Dist.begin(), nearbyP_Dist.end());
						int  nearestP_Idx = nearbyP_Idx[distance(nearbyP_Dist.begin(), nearest)];

						Rect pRect = pCandidates[nearestP_Idx]->curRect;

						// Determine attach height of ball (bc->attachedHeight)
						updateAttachedHeight(pRect, bc);

						// Create large window
						bc->curCrd = Point(pRect.x + pRect.width/2, pRect.y + int(bc->attachedHeight * pRect.height));
						#ifdef WINDOW_PERSPECTIVE
							bc->curRad = Point(pRect.width/2, pRect.height/2) + perspectiveRad(attachRad, bc->curCrd);
						#else
							bc->curRad = Point(pRect.width/2, pRect.height/2) + attachRad;
						#endif
						bc->fitFrame();

						bc->curAppearM = 0.0;
						bc->switchState(curFrame, BALL_STATE::ATTACHED_TO_PLAYER);
						break;
					}			
					
					/**********************************************************
					No nearest player - Continue. Correlate and search for best match
					***********************************************************/
					
					vector<Point> matchPoints;
					vector<double> matchProbs;
					appearAnalyzer.getMatches(bc, 1, matchPoints, matchProbs, TID);

					// matchPoints is never empty
					if (matchPoints.empty()) 
					{
						bc->switchState(curFrame, BALL_STATE::GOT_LOST);
						break;
					}

					Point nCrd   = matchPoints[0];
					double nProb = matchProbs[0];

					//bc->predCrd = bc->KF.correct(nCrd);								// UPDATE FRAME T
					//bc->predCrd = Point(bc->KF.predict().x, bc->KF.predict().y);	// PREDICT FOR FRAME T+1
					//bc->predCrd = Point(bc->KF.predict().x, bc->KF.predict().y);
					
					if (nProb < M1_loose_threshold) 
					{
						bc->switchState(curFrame, BALL_STATE::SEARCHING);
						break;
					}

					// --- no switch performed
					bc->curCrd = nCrd;
					bc->curAppearM = nProb;
					break;
				}
				//_____________________________________________________________
				case BALL_STATE::SEARCHING : {
				//*************************************************************
					if (bc->getLastStateDuration(curFrame) > 4) 
					{
						bc->switchState(curFrame, BALL_STATE::GOT_LOST);
						break;
					}

					// Increase window size
					#ifdef WINDOW_PERSPECTIVE
						bc->curRad = bc->curRad + perspectiveRad(searchIncRad, bc->curCrd);
					#else
						bc->curRad = bc->curRad + searchIncRad;
					#endif						
					bc->fitFrame();

					vector<Point> matchPoints;
					vector<double> matchProbs;

					/**********************************************************
								Correlate and search for best match
					***********************************************************/
					appearAnalyzer.getMatches(bc, 1, matchPoints, matchProbs, TID);
					
					Point nCrd   = matchPoints[0];
					double nProb = matchProbs[0];

					// Match found
					if (nProb > M1_find_threshold) 
					{
						bc->curCrd		= nCrd;
						bc->curAppearM	= nProb;

						// Set window size to default
						#ifdef WINDOW_PERSPECTIVE
							bc->curRad = perspectiveRad(defRad, bc->curCrd);
						#else
							bc->curRad = defRad;
						#endif
						bc->fitFrame();

						bc->switchState(curFrame, BALL_STATE::TRACKING);
						break;
					}
					break;
				}
				//_____________________________________________________________
				case BALL_STATE::GOT_LOST : {
				//*************************************************************
					// do nothing
					break;
				}
				//_____________________________________________________________
				case BALL_STATE::ATTACHED_TO_PLAYER : {
				//*************************************************************
					
					vector<double> nearbyP_Dist;
					vector<int> nearbyP_Idx;

					/**********************************************************
										Find Nearest Player
					***********************************************************/
					getNearestPlayersVctr(bc, nearbyP_Idx, nearbyP_Dist);
					
					if (!nearbyP_Idx.empty()) 
					{
						int minIdx = distance(nearbyP_Dist.begin(), min_element(nearbyP_Dist.begin(), nearbyP_Dist.end()));
						int nearestPlayerIdx = nearbyP_Idx[minIdx];
						double nearestPlayerDist = nearbyP_Dist[minIdx];

						// Locate nearest player
						if (nearestPlayerDist < 15.0)
						{
							// Move window to the player
							Rect pRect = pCandidates[nearestPlayerIdx]->curRect;
							bc->curCrd = Point(pRect.x + pRect.width/2, pRect.y + int(bc->attachedHeight * pRect.height));

							// Create large window
							#ifdef WINDOW_PERSPECTIVE
								bc->curRad = Point(pRect.width/2, pRect.height/2) + perspectiveRad(attachRad, bc->curCrd);
							#else
								bc->curRad = Point(pRect.width/2, pRect.height/2) + attachRad;
							#endif
							bc->fitFrame();
							bc->curAppearM = 0.0;

							/**********************************************************
										Prepare and perform template matching
							***********************************************************/
							// ------ 1) create mask from all nearby players
							Rect winRect = bc->curRect;
							Mat mask = Mat(winRect.height, winRect.width, CV_8UC3, Scalar(1,1,1)); // Original (1,1,1)
							for (auto np : nearbyP_Idx) 
							{
								Rect eachRect = (pCandidates[np]->curRect) & winRect;
								if (eachRect == Rect()) 
								{
									continue;
								}

								Mat roi = mask(eachRect-Point(winRect.x, winRect.y));
								Mat mask_0 = Mat(eachRect.height, eachRect.width, CV_8UC3, Scalar(0,0,0)); // Original (0,0,0)
								mask_0.copyTo(roi);
							}

							// ------ 2) matchTemplate for region except mask
							bc->restrictedMask = mask;

							vector<Point> matchPoints;
							vector<double> matchProbs;

							// Correlate.. Task -> include distance constraint ( Use filter )
							appearAnalyzer.getMatches(bc, 2, matchPoints, matchProbs, TID);

							vector<Point>  nCrds = matchPoints;
							vector<double> nProbs = matchProbs;

							/**********************************************************
												Search for best match
							***********************************************************/
							for (int i = 0; i < nCrds.size(); i++)
							{
								vector<double> nearbyP_Dist;
								vector<int> nearbyP_Idx;

								// Calculate distance of match to all players
								getNearestPlayersVctr(nCrds[i], nearbyP_Idx, nearbyP_Dist);

								if (!nearbyP_Idx.empty())
								{
									int minIdx = distance(nearbyP_Dist.begin(), min_element(nearbyP_Dist.begin(), nearbyP_Dist.end()));
									int nearestPlayerIdx = nearbyP_Idx[minIdx];
									double nearestPlayerDist = nearbyP_Dist[minIdx];

									// Safest match if candidate is isolated
									if (nearestPlayerDist > 50.0 && nProbs[i] > 0.95)
									{
										bc->curCrd = nCrds[i];
										bc->curAppearM = nProbs[i];

										// Revert to default window
										#ifdef WINDOW_PERSPECTIVE
										bc->curRad = perspectiveRad(defRad, bc->curCrd);
										#else
										bc->curRad = defRad;
										#endif
										bc->fitFrame();

										bc->switchState(curFrame, BALL_STATE::TRACKING);
										break;
									}
								}
							}

							// Safest match not found - Designate the highest match as the ball
							if (nProbs[0] > 0.96)
							{
								bc->curCrd = nCrds[0];
								bc->curAppearM = nProbs[0];
								
								// Keep large window
								#ifdef WINDOW_PERSPECTIVE
								bc->curRad = perspectiveRad(defRad, bc->curCrd);
								#else
								bc->curRad = defRad;
								#endif
								bc->fitFrame();
								
								bc->switchState(curFrame, BALL_STATE::TRACKING);
								break;
							}

							bc->curAppearM = nProbs[0];
							//bc->switchState(curFrame, BALL_STATE::TRACKING);

							break;
						}
					}
					
					#ifdef WINDOW_PERSPECTIVE
						bc->curRad = perspectiveRad(defRad, bc->curCrd);
					#else
						bc->curRad = defRad;
					#endif // WINDOW_PERSPECTIVE
					bc->fitFrame();
					bc->switchState(curFrame, BALL_STATE::TRACKING);
					break;
				}
				//_____________________________________________________________
				case BALL_STATE::SEPARATED_FROM_PLAYER : {
				//*************************************************************
					break;
				}
				//_____________________________________________________________
				case BALL_STATE::KICKED : {
				//*************************************************************
					break;
				}
			}
			
			// Update information of Ball Candidate
			bc->updateStep();
		}

		//=========================================================================================
		void ball_updateCandGroups () {
			bCandidatesGroups.clear();
			bCandidatesGroups = vector<vector<BallCandidate*>>(BALL_STATE::BALL_STATES_COUNT);

			// Separates all the Ball Candidates into their respective groups [ SEARCHING / TRACKING / ETC ]
			for (auto c : bCandidates) 
			{
				int state = c->getState();
				bCandidatesGroups[state].push_back(c);
			}
		}

		//=========================================================================================
		void ball_chooseMainCandidate () {
			ball_updateCandGroups();

			// ---------- if there is at least one tracking candidates, remove all non-tracking ----------
			if (bCandidatesGroups[BALL_STATE::TRACKING].size() > 0) 
			{
				vector<BallCandidate*> mCandidates = bCandidatesGroups[BALL_STATE::TRACKING];
				
				// mainCandidate is the candidate with the highest CC score
				sort(mCandidates.begin(), mCandidates.end(), BallCandidate::compareLastAppearM);
				mainCandidate = mCandidates[0];

				// Remove the rest
				vector<BallCandidate*> toDelete;
				for (auto c : bCandidates) 
				{
					if (c != mainCandidate) 
					{
						toDelete.push_back(c);
					}
				}

				checkMainCandBeforeDelete(toDelete);
				deleteTrackObjects_(bCandidates, toDelete);
			}
		}

		//=========================================================================================
		void ball_removeLostCandidates () {

			vector<BallCandidate*> toDelete;

			for (auto bc : bCandidates)
			{
				BALL_STATE st = bc->getState();
				if (st == BALL_STATE::GOT_LOST || st == BALL_STATE::OUT_OF_FIELD)
				{
					toDelete.push_back(bc);
				}
			}

			checkMainCandBeforeDelete(toDelete);
			deleteTrackObjects_(bCandidates, toDelete);
		}

		//=========================================================================================
		void ball_removeOutdatedCandidates (int maxLifeTime) {

			vector<BallCandidate*> toDelete;

			for (auto bc : bCandidates)
			{
				if (bc->lifeTime > maxLifeTime) 
				{
					toDelete.push_back(bc);
				}
			}

			checkMainCandBeforeDelete(toDelete);
			deleteTrackObjects_(bCandidates, toDelete);
		}

		//=========================================================================================
		void ball_removeStuckedCandidates (double ratio) {

			vector<BallCandidate*> toDelete;
			vector<BALL_STATE> states;

			states.push_back(BALL_STATE::INIT);
			states.push_back(BALL_STATE::TRACKING);

			for (auto bc : bCandidates) 
			{
				// track Ratio is a function of the duration at which the ball has being tracked
				double trackRatio = double(bc->getStateDuration(states)) / bc->lifeTime;
				if (trackRatio < ratio) 
				{
					toDelete.push_back(bc);
				}
			}

			checkMainCandBeforeDelete(toDelete);
			deleteTrackObjects_(bCandidates, toDelete);
		}

		//=========================================================================================
		void trackPlayers (vector<Rect>& newCandidates, Mat& frame, int TID, Mat mask) {
			static int count = 0;

			appearAnalyzer.setFrame(frame);

			// calculate coords for all new candidates
			vector<Point> newCandCoords;
			for (auto& c : newCandidates)
			{
				newCandCoords.push_back(Point (c.x + c.width/2, c.y + c.height));
			}

			// proceed all existing players
			vector<int> usedCandidates (newCandCoords.size(), 0);
			double minDist = 20.0;

			for (auto p : pCandidates) 
			{
				// ----- calulate distance to all new candidates
				vector<double> distance(newCandidates.size());
				getDist(p->curCrd, newCandCoords, distance, true);
				int minIdx = std::distance(distance.begin(), min_element(distance.begin(), distance.end()));
				
				// If new player candidate is inside existing candidate
				if ((!newCandidates.empty()) && (distance[minIdx] < minDist)) 
				{
					// --- choose the closest one
					Point nCrd = newCandCoords[minIdx];
					
					// --- Kalman Filter
					#ifdef PLAYERS_KF
					p->KF.predict(); // Predict based on last position

					Point cc = p->KF.correct(nCrd); // Correct based on prediction and measurement (nCrd)

					p->setCrd(cc, false);
					#else
					p->setCrd(nCrd, false);
					#endif

					// Set newly detected rectangle as curRect
					Rect nRect = newCandidates[minIdx];
					int pCandidate_teamID = appearAnalyzer.getTeamID(newCandidates[minIdx]);
					p->teamID = pCandidate_teamID;
					p->setRect(nRect);

					p->Occlusion = true;

					// Candidate is used
					int idx = std::distance(newCandCoords.begin(), std::find(newCandCoords.begin(), newCandCoords.end(), nCrd));
					usedCandidates[idx] = 1;
				} 
				
				else 
				{
					// --- or predict new position
					#ifdef PLAYERS_KF
					Point nCrd = p->KF.predict();
					#else
					Point nCrd = p->curCrd;
					#endif // PLAYERS_KF
					p->setCrd(nCrd, true);

					p->Occlusion = true;

					// Set previously detected rectangle as curRect
					p->setRect(nCrd);
				}
			}

			// ----- merge players -----
			mergePlayers(frame);

			// ----- add new players -----
			for (unsigned i = 0; i < usedCandidates.size(); i++)
			{
				// Newly detected candidates
				if (usedCandidates[i] == 0)
				{
					// Get team
					int pCandidate_teamID = appearAnalyzer.getTeamID(newCandidates[i]);

					// Construct player obj
					pCandidates.push_back(new PlayerCandidate(curFrame, newCandCoords[i], newCandidates[i], appearAnalyzer.getTeamID(newCandidates[i]), true));
				}
			}
			
			// ----- Identify players under occlusion
			/*for (int i = 0; i < pCandidates.size(); i++)
			{
				auto cand1 = pCandidates[i];

				vector<PlayerCandidate*> playersOccluded;
				vector<Rect> playersRect; 		
				vector<int> usedID;

				playersOccluded.push_back(cand1);
				playersRect.push_back(cand1->curRect);
				usedID.push_back(i);

				// Loop through other players
				for (int j = 0; j < pCandidates.size(); j++)
				{
					// Same player or not enough data to process
					if (i == j || pCandidates[j]->prevRects.size() < 2) continue;

					auto cand2 = pCandidates[j];
					Point cand2Crd = Point(cand2->curCrd.x, cand2->curCrd.y - cand2->curRect.height / 2);

					// Center of other player is inside of current player and their areas differ significantly
					if (cand1->curRect.contains(cand2Crd) && cand1->curRect.area() > 1.7 * cand2->curRect.area())
					{
						playersOccluded.push_back(pCandidates[j]);
						playersRect.push_back(pCandidates[j]->curRect);
						usedID.push_back(j);
					}
				}

				// Solve for Occlusion
				if (playersOccluded.size() > 1)
					appearAnalyzer.segmentPlayer(playersOccluded, mergeBoundingBoxes(playersRect), frame, TID, mask);
			}*/

			// ----- update info for all players -----
			for (auto p : pCandidates) 
			{
				p->updateStep();
			}

			// ----- delete outdated players -----
			vector<PlayerCandidate*> toDelete;

			for (auto p : pCandidates) 
			{
				if (p->predictTime > 5) 
				{
					toDelete.push_back(p);
				}
			}
			
			deleteTrackObjects_(pCandidates, toDelete);
		}

		//=========================================================================================
																  /*nearbyP_Idx,          nearbyP_Dist*/
		void getNearestPlayersVctr (BallCandidate* bc, vector<int>& pIndexes, vector<double>& pDistance, double maxDist = numeric_limits<double>::max(), vector<int>& pID = vector<int>()) {
			
			// !!! add priority
			pIndexes.clear();
			pDistance.clear();

			vector<Rect> pRects(pCandidates.size());
			vector<int> _pID(pCandidates.size());
			vector<double> pDistance_all(pCandidates.size());

			// pRect = container for pCandidate
			for (unsigned i = 0; i < pCandidates.size(); i++) 
			{
				pRects[i] = pCandidates[i]->curRect;
				_pID[i] = pCandidates[i]->id;
			}

			// Get distance from current candidate to all player rectangles
			getDist(bc->curCrd, pRects, pDistance_all);

			for (unsigned i = 0; i < pRects.size(); ++i)
			{
				// Update distance if not out of bounds
				if (pDistance_all[i] <= maxDist)
				{
					pDistance.push_back(pDistance_all[i]);
					pIndexes.push_back(i);
					pID.push_back(_pID[i]);
				}
			}
		}

		//=========================================================================================
		void getNearestPlayersVctr(Point match, vector<int>& pIndexes, vector<double>& pDistance, double maxDist = numeric_limits<double>::max()) {

			// !!! add priority
			pIndexes.clear();
			pDistance.clear();

			vector<Rect> pRects(pCandidates.size());
			vector<double> pDistance_all(pCandidates.size());

			for (unsigned i = 0; i < pCandidates.size(); i++)
			{
				pRects[i] = pCandidates[i]->curRect;
			}

			getDist(match, pRects, pDistance_all);
			
			for (unsigned i = 0; i < pRects.size(); ++i)
			{
				if (pDistance_all[i] <= maxDist)
				{
					pDistance.push_back(pDistance_all[i]);
					pIndexes.push_back(i);
				}
			}
		}

		//=========================================================================================
		inline int players_searchByID (int _id) {
			int res = -1;
			for (unsigned i = 0; i < pCandidates.size(); ++i) {
				if (pCandidates[i]->id == _id) {
					res = i;
					break;
				}
			}
			return res;
		}

		//=========================================================================================
		inline void getDist (Point& a, vector<Point>& vctr, vector<double>& dist, bool _sqrt = false) {
			
			if (!_sqrt) 
			{
				for (unsigned i = 0; i < vctr.size(); i++) 
					dist[i] = (vctr[i].x-a.x)*(vctr[i].x-a.x) + (vctr[i].y-a.y)*(vctr[i].y-a.y);

			} 
			
			else 
			{
				for (unsigned i = 0; i < vctr.size(); i++)
					dist[i] = sqrt((vctr[i].x-a.x)*(vctr[i].x-a.x) + (vctr[i].y-a.y)*(vctr[i].y-a.y));
			}
		}
		
		//=========================================================================================
		inline void getDist (Point& a, vector<Rect>& vctr, vector<double>& dist, bool _sqrt = false) {
			
			Rect r;
			double d;

			for (unsigned i = 0; i < vctr.size(); i++)
			{
				r = vctr[i];

				// the distance has to be calculated as the minimal distance to rect corners
				if ((a.x < r.x) && (a.y<r.y)) {
					d = (a.x - r.x) * (a.x-r.x) + (a.y - r.y) * (a.y - r.y);
				} else if ((a.x > r.x + r.width) && (a.y < r.y)) {
					d = (a.x - r.x - r.width) * (a.x - r.x - r.width) + (a.y - r.y) * (a.y - r.y);
				} else if ((a.x < r.x) && (a.y > r.y + r.height)) {
					d = (a.x - r.x) * (a.x - r.x) + (a.y - r.y - r.height) * (a.y - r.y - r.height);
				} else if ((a.x > r.x + r.width) && (a.y > r.y + r.height)) {
					d = (a.x - r.x - r.width) * (a.x - r.x - r.width) + (a.y - r.y - r.height) * (a.y - r.y - r.height);
				}

				// the distance has to be calculated as the minimal distance to rect sides
				else if (a.y < r.y) {
					d = (a.y - r.y) * (a.y - r.y);
				} else if (a.x < r.x) {
					d = (a.x - r.x) * (a.x - r.x);
				} else if (a.y > r.y + r.height) {
					d = (a.y - r.y - r.height) * (a.y - r.y - r.height);
				} else if (a.x > r.x + r.width) {
					d = (a.x - r.x - r.width) * (a.x - r.x - r.width);
				}

				// the point is inside rect
				else {
					d = 0;
				}
				if (_sqrt) {
					d = sqrt(d);
				}

				dist[i] = d;
			}
		}

		//=========================================================================================
		void checkMainCandBeforeDelete (vector<BallCandidate*>& toDelete) {

			for (auto& c : toDelete) 
			{
				if (c == mainCandidate) 
				{
					mainCandidate = NULL;
				}
			}
		}

		//=========================================================================================
		template <class objType>
		inline void deleteTrackObjects_ (vector<objType*>& collection, vector<objType*>& toDelete) {

			for (auto& obj : toDelete) 
			{
				collection.erase(remove(collection.begin(), collection.end(), obj), collection.end());
				delete obj;
			}

			toDelete.clear();
		}

		//=========================================================================================
		void ball_addMoreCandidates(vector<Point>& possibleCandidates, Mat& frame, int cnt, int TID, Rect restrictedZone = Rect()) {

			vector<BallCandidate*> tCandidates;

			for (auto possCand : possibleCandidates)
			{
				bool contains = false;
				// ----- check that new candidate is not inside existent candidates -----
				for (auto existCand : bCandidates)
				{
					if (existCand->curRect.contains(possCand))
					{
						contains = true;
						break;
					}
				}

				// ----- check that new candidate is not inside existent players -----
				for (auto player : pCandidates)
				{
					if (player->curRect.contains(possCand))
					{
						contains = true;
						break;
					}
				}

				if (!contains)
				{
					// Update tCandidates
					tCandidates.push_back(new BallCandidate(curFrame, possCand, iniRad, 0.0));
				}
			}

			appearAnalyzer.setFrame(frame);

			// Calculate score for tCandidates
			for (auto possCand : tCandidates)
			{
				vector<Point> points;
				vector<double> probs;

				// Correlate
				appearAnalyzer.getMatches(possCand, 1, points, probs, TID);

				Point nCrd = possCand->curCrd;
				double nProb = 0.0;

				if (!points.empty())
				{
					nCrd = points[0];
					nProb = probs[0];
				}

				possCand->curCrd = nCrd;

				#ifdef WINDOW_PERSPECTIVE
					possCand->curRad = perspectiveRad(defRad, possCand->curCrd);
				#else
					possCand->curRad = defRad;
				#endif // WINDOW_PERSPECTIVE

				possCand->curAppearM = nProb;
				possCand->updateStep();
			}

			// From tCandidates, pick the good ones ( can be > 1 ) to be stored into ballCandidates or bc
			sort(tCandidates.begin(), tCandidates.end(), BallCandidate::compareLastAppearM);
			vector<BallCandidate*> toDelete;
			int lh = int(min(double(cnt), double(tCandidates.size())));
			
			for (int i = 0; i < lh; i++)
			{
				BallCandidate* bc = tCandidates[i];

				// If score > 0.9, good ball candidate detected, begin tracking
				if (bc->curAppearM > 0.9)
				{
					bc->switchState(curFrame, BALL_STATE::TRACKING);
					bCandidates.push_back(bc);
				}

				// If score < 0.9, remove
				else
				{
					toDelete.push_back(tCandidates[i]);
				}
			}

			deleteTrackObjects_(tCandidates, toDelete);
		}

		//=========================================================================================
		void drawTrackingMarks (Mat& frame, int TID) {

			// draw ball candidates trace
			for (auto bc : bCandidates) 
			{
				drawTrackingMarks(frame, bc, TID);
			}

			// draw player candidates trace
			for (auto c : pCandidates) 
			{
				unsigned s = unsigned(max(double(c->coordsKF.size())-50, 1.0));
				for (unsigned i = s; i < c->coordsKF.size(); i++)
				{
					Point x = c->coordsKF[i - 1];
					Point y = c->coordsKF[i];

					if		(c->teamID == 1) line(frame, x, y, CV_RGB(0, 0, 255), 1, CV_AA);
					else if (c->teamID == 0) line(frame, x, y, CV_RGB(255, 255, 255), 1, CV_AA);
					else if (c->teamID == 2) line(frame, x, y, CV_RGB(0, 0, 0), 1, CV_AA);
					else					 line(frame, x, y, CV_RGB(128, 50, 0), 1, CV_AA);
				}
			}

			// draw rectangles around the players
			for (auto p : pCandidates) 
			{
				if (p->prevRects.size() > 1)
				switch (p->teamID) {

				case 0: { rectangle(frame, p->curRect, CV_RGB(255, 255, 255), 1); break; } // White Team
				case 1: { rectangle(frame, p->curRect, CV_RGB(0, 0, 255), 1); break; } // Blue Team
				case 2: { rectangle(frame, p->curRect, CV_RGB(0, 0, 0)); break; } // Referee
				default: { rectangle(frame, p->curRect, CV_RGB(128, 50, 0)); break; } // Unknown

				}

				/*char text[40];
				sprintf(text, "%d", p->Occlusion);
				putText(frame, text, Point(p->curCrd.x, p->curCrd.y - 20*p->teamID), CV_FONT_HERSHEY_SIMPLEX, 2, CV_RGB(255*p->teamID, 0, 0), 1);*/
			}

			// ----- display number of players tracked and number of ballCandidates -----
			string label = "";
			//label += to_string(pCandidates.size());
			
			//if (TID == 3 && bCandidates.size() != 0)
			//{
			//	static int count = 0;
			//	Rect roi = Rect(Point(bCandidates[0]->curCrd - Point(120, 100)), Point(bCandidates[0]->curCrd + Point(120, 100))) & Rect(0,0,960,540);

			//	
			//	Point balance = Point(240 - roi.width, 200 - roi.height);
			//	

			//	Rect actual = Rect(roi.x, roi.tl().y - balance.y, 240/*roi.br().x + balance.x*/, 200/*roi.br().y*/);
			//	//Rect actual = Rect(Point(roi.tl() - balance), Point(roi.br())) & Rect(0,0,960,540);
			//	
			//	Mat show = frame(actual);
			//	char filename[40];
			//	sprintf(filename, "Track_%d.png", count);
			//	imwrite(filename, show);
			//	count++;
			//	imshow("L", show);
			//	waitKey(1);
			//}

			//label += /*";  " + */metric.toString_prc();
			//setLabel (frame, Rect(0, 0, 900, 20), label);
		}

		//=========================================================================================

		void drawTrackingMarks (Mat& frame, BallCandidate* bc, int TID) {

			if (bc != mainCandidate || mainCandidate->getState() == 1) 
			{
				return;
			}

			// draw track marks of mainCandidate
			unsigned s = unsigned(max(double(bc->coords.size())-25, 1.0));
			for (unsigned i = s; i < bc->coords.size(); i++) 
			{
				line(frame, bc->coordsKF[i-1], bc->coordsKF[i], CV_RGB(255, 0, 0), 1, CV_AA);
			}

			// Ball state
			Scalar boxColor = CV_RGB(0,0,0);
			switch (bc->getState()) {
			
			case BALL_STATE::TRACKING :
				boxColor = CV_RGB(255, 255, 255); // White
				break;
			
			case BALL_STATE::SEARCHING :
				boxColor = CV_RGB(255, 255, 0);   // Yellow
				break;
			
			case BALL_STATE::ATTACHED_TO_PLAYER :
				boxColor = CV_RGB(255, 0, 0);     // Red
				break;
			}

			/********************************************************************************
									Display information on camera view
			*********************************************************************************/

			// Track ratio
			//double trackRatio = double(bc->getStateDuration(BALL_STATE::TRACKING)) / bc->lifeTime;

			Rect box = Rect(Point(bc->curCrd.x - 20, bc->curCrd.y - 20), Point(bc->curCrd.x + 20, bc->curCrd.y + 20));

			//// Ball state
			rectangle(frame, box, CV_RGB(255, 0, 0)); // bc->curRect

			////Height of ball ( but frame t - 1 )
			float height = 0;
			for (int i = 0; i < Ball.size(); i++)	if (TID == Ball[i]->cameraID)	if (Ball[i]->coords3D.size() != 0) height = (Ball[i]->coords3D.end() - 1)->z;

			if (height == 0) return;

			// Display text
			//char label[40];
			//sprintf(label, "%2.2f", height/*, bc->id, Ball.size()*//*, bc->curAppearM, Ball.size()*/);
			//putText(frame, label, Point((bc->coordsKF.end() - 1)->x + 40, (bc->coordsKF.end() - 1)->y + 20), CV_FONT_HERSHEY_SIMPLEX, 3, CV_RGB(255, 0, 0), 2);

		}

		//=========================================================================================
		inline void setLabel(Mat& im, Rect r, const string label, Scalar backColor=CV_RGB(0,0,0), Scalar textColor=CV_RGB(255,255,255)) {
			int fontface = cv::FONT_HERSHEY_DUPLEX;
			double scale = 1.0;
			int thickness = 1;
			int baseline = 0;

			Size text = getTextSize(label, fontface, scale, thickness, &baseline);
			Point pt(r.x + (r.width-text.width)/2, r.y + (r.height+text.height)/2);
			
			rectangle(im, pt + Point(0, baseline), pt + Point(text.width, -text.height), backColor, CV_FILLED);
			putText(im, label, pt, fontface, scale, textColor, thickness, 8);
		}

		//=========================================================================================
		template < class objType, class fType >
		set<set<objType*>> merge_(vector<objType*> items, fType compareMerge) {

			int s = items.size();
			if (s == 0) 
			{
				return set<set<objType*>>();
			}

			vector<pair<objType*, objType*>> pairs;
			set<set<objType*>> groups;

			// --- allocate table ---
			// Each player[i] is assigned remaining players[j]
			bool** rel = new bool*[s];
			for (int i = 0; i < s; ++i) 
			{
				rel[i] = new bool[s];
				for (int j = 0; j < s; j++) 
				{
					rel[i][j] = false;
				}
			}

			// --- fill the table, create pairs
			for (int i = 0; i < s; i++) 
			{
				for (int j = 0; j < s; j++) 
				{
					if (i == j) { continue;	}

					// if the bounding boxes or i and j intersect ?
					if (compareMerge(items[i], items[j]) && (rel[j][i] == false)) 
					{
						rel[i][j] = true;
						pairs.push_back(pair<objType*, objType*>(items[i],items[j]));
					}
				}
			}

			// merge pairs into groups
			for (auto pr : pairs) 
			{
				objType* t1 = pr.first;
				objType* t2 = pr.second;

				bool inserted = false;

				for (auto i : groups) 
				{
					if (i.find(t1) != i.end()) 
					{
						i.insert(t2);
						inserted = true;
						break;
					} 

					else if (i.find(t2) != i.end()) 
					{
						i.insert(t1);
						inserted = true;
						break;
					}
				}

				if (!inserted) 
				{
					set<objType*> ts;
					ts.insert(t1);
					ts.insert(t2);
					groups.insert(ts);
				}
			}

			// --- delete table ---
			for (int i = 0; i < s; ++i) {
				delete [] rel[i];
			}

			delete [] rel;
			return groups;
		}

		//=========================================================================================
		void mergeWindows () {

			auto groups = merge_(bCandidates, &Tracker::compareMerge_Window);
			
			vector<BallCandidate*> toDelete;
			for (auto i : groups) 
			{
				if (i.empty()) 
				{
					break;
				}

				Rect rectIntersection = (*i.begin())->curRect, rectUnion = (*i.begin())->curRect;
				double mergedProb = 0.0;
				
				for (auto j : i) 
				{
					rectIntersection = rectIntersection & j->curRect;
					rectUnion = rectUnion | j->curRect;
					mergedProb = max(mergedProb, j->curAppearM);
					toDelete.push_back(j);
				}

				Point mergedCrd = Point(rectIntersection.x + rectIntersection.width/2, rectIntersection.y + rectIntersection.height/2);

				Point mergedRad (rectUnion.width/2, rectUnion.height/2);
				BallCandidate* nbc = new BallCandidate(curFrame, mergedCrd, mergedRad, mergedProb);
				// !!!!!!!!!!!!!!!
				nbc->switchState(curFrame, BALL_STATE::SEARCHING);
				bCandidates.push_back(nbc);
			}

			deleteTrackObjects_(bCandidates, toDelete);
		}
			
		//=========================================================================================
		void mergePlayers(Mat& frame) {

			// Extract pair of candidates who share the same coordinates and are not under occlusion
			auto groups = merge_(pCandidates, &Tracker::compareMerge_Player);
			
			vector<PlayerCandidate*> toDelete;
			
			// Let one of the pair be the candidate
			for (auto i : groups) 
			{
				if (i.empty()) break;

				Point mergedCrd;
				Rect mergedRect;
				int mergedID;

				for (auto j : i) 
				{
					mergedCrd = j->curCrd;
					mergedRect = j->curRect;
					mergedID = j->teamID;
					toDelete.push_back(j);
				}

				pCandidates.push_back(new PlayerCandidate(curFrame, mergedCrd, mergedRect, mergedID, true));
			}

			// Delete one of the pairs
			deleteTrackObjects_(pCandidates, toDelete);
		}

		Rect mergeBoundingBoxes(vector<Rect> rects) {

			int xmin = min(rects[0].x, rects[1].x);
			int ymin = min(rects[0].y, rects[1].y);
			int xmax = max(rects[0].x + rects[0].width, rects[1].x + rects[1].width);
			int ymax = max(rects[0].y + rects[0].height, rects[1].y + rects[1].height);

			return Rect(xmin, ymin, xmax - xmin, ymax - ymin);
		}

		//=========================================================================================
		Point Inv_Triangulate(Point3f cam, Point3f ball) {

			// Vector of line from camera to ball
			Point3f vector = ball - cam;

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
		vector<pair<float,Point>> evaluateContours(Mat roi, Point tl)	{
			
			vector<pair<float,Point>> candidate;

			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			
			// Image processing
			cvtColor(roi, roi, CV_BGR2GRAY);
			threshold(roi, roi, 128, 255, THRESH_BINARY);

			// Contour extraction
			findContours(roi, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

			// Contour Analysis
			for (auto c : contours) if (contourArea(c) > 0.01)
			{
				// Compute circularity
				float circularity = pow(arcLength(c, true), 2) / (12.5664*contourArea(c));

				// Compute center
				Rect br = boundingRect(Mat(c));
				Point centre(br.x + br.width / 2, br.y + br.height / 2);

				// Update candidate if circularity < 2
				if(circularity < 2.0) candidate.push_back(make_pair(circularity, centre + tl));
			}		

			return candidate;
		}

		//=========================================================================================
		void temporaryFunction1(Point mid_roi, int length, int width, Point contourCentre) {

			Point tl(mid_roi - Point(length / 2, width / 2));
			contourCentre = contourCentre + tl;
		}

		//=========================================================================================
		~Tracker(void) {
			std::cout << "tracker is destructing" << std::endl;
			for (unsigned i = 0; i < bCandidates.size(); i++) {
					delete bCandidates[i];
			}

		}
};


}