#pragma once

#include <cv.h>
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
		void initialize () {

			appearAnalyzer.setBallTempls(ballTempls);
			appearAnalyzer.generateClassifier();

			curFrame = 0;
			flg = true;

			defRad       = Point(45, 35);  // Initial search radius (45, 35)
			iniRad       = Point(22, 22);  // (?)  
			attachRad    = Point(100, 60);
			searchIncRad = Point(15,15);   // Rate at which size of search box increases

			M1_loose_threshold = 0.94;
			M1_find_threshold = 0.96;

			restrictedArea = Mat(fSize, CV_32FC1, 1.0);
			restrictedArea(Rect(0, 0, fSize.x, int(0.08*fSize.y))) = 0.0;
			restrictedArea(Rect(0, int(0.93*fSize.y), fSize.x, int(0.07*fSize.y))) = 0.0;
			appearAnalyzer.setRestrictedArea(restrictedArea);

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
				trackInfo.set(-1, Rect(), Point(-1, -1), Point(), givenTrajectory[curFrame + 2], metric.results);
			} 
			
			else 
			{
				trackInfo.set(mainCandidate->id, mainCandidate->curRect, mainCandidate->curCrd, mainCandidate->predCrd, givenTrajectory[curFrame + 2], metric.results);
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
		inline void updateMetric(ofstream& file) {
			Point p = givenTrajectory[curFrame + 3];

			bool exists = (p != outTrajPoint);
			bool detected = (mainCandidate != NULL);			

			if (exists)
			{	
				metric.Total++;

				// Candidate found
				if (detected) 
				{
					// Save results to file
					file << p.x << " " << p.y << " " << mainCandidate->curCrd.x << " " << mainCandidate->curCrd.y << endl;

					// Compute euclidean distance of candidate to GT
					double dist = sqrt(distSQ(mainCandidate->curCrd, p));

					// Update track ratio of main candidate
					if (mainCandidate->getState() == BALL_STATE::TRACKING) { metric.addDistance(dist); metric.TR = double(mainCandidate->getStateDuration(BALL_STATE::TRACKING)) / mainCandidate->lifeTime; }
					else												     metric.addDistance(0.0);
					
					if (dist < 5) 	metric.TP++; // correctly detected
					else       	    metric.TN++; // incorrectly detected

					// Update vector of tracking results
					metric.results.push_back(make_pair(p, mainCandidate->curCrd));
				} 
				
				// Candidate not found
				else 
				{
					// Save results to file
					file << p.x << " " << p.y << " " << -1 << " " << -1 << endl;

					metric.addDistance(0.0);
					metric.TN++;

					// Update vector of tracking results
					metric.results.push_back(make_pair(p, Point(-1,-1)));
				}

			} 
			
			// The real ball doesn't exist
			else 
			{
				// Save results to file
				file << p.x << " " << p.y << " " << -1 << " " << -1 << endl;

				metric.addDistance(0.0);
				
				if (detected)	metric.FP++; // incorrectly detected
				else 			metric.FN++; // no candidate found
				
				// Update vector of tracking results. Task -> continue to update mainCandidate->curCrd
				metric.results.push_back(make_pair(p, p));
			}
		}

		//=========================================================================================
		void processFrame(Mat& frame, vector<Point>& ball_cand, vector<Rect>& player_cand, vector<ProjCandidate*> Ball, int TID, int processedFrames, ofstream& file) {

			this->Ball = Ball;

			trackPlayers(player_cand, frame);
			trackBall(frame, ball_cand, Ball, TID, processedFrames);

			drawTrajectory(frame, 2);
			updateMetric(file);

			drawBallGroundTruth(frame, 5);
			//drawDistance(frame);

			curFrame++;
		}

		//=========================================================================================
		static bool compareMerge_Player (PlayerCandidate* c1, PlayerCandidate* c2) {
			// --- check if two players need to be merged
			return (c1->curCrd == c2->curCrd);
		}
		
		//=========================================================================================
		static bool compareMerge_Window (BallCandidate* c1, BallCandidate* c2) {
			// --- check if two ball candidates need to be merged
			Rect rectInersection = c1->curRect & c2->curRect;
			if (rectInersection.area() > 0.7*min(c1->curRect.area(), c2->curRect.area())) {
				return true; // merge needed
			}
			return false;
		}
		//=========================================================================================
		void ballOutOfPlay() {
		
}

		//=========================================================================================
		void trackBall(Mat& frame, vector<Point>& newCandidates, vector<ProjCandidate*> truePositives, int TID, int processedFrames) {
	
			appearAnalyzer.setFrame(frame); 

			switch (trackerState) {

				//_____________________________________________________________
				case TRACKER_STATE::BALL_NOT_FOUND :


					// Need condition.. If prev trajectory indicates that ball is out of play.. Switch to 1
					// But when to switch back to zero ?

					/*******************************************************************

												BALL OUT OF PLAY

					********************************************************************/					
					for (unsigned i = 0; i < pCandidates.size(); i++)
					{
						int feet = pCandidates[i]->curRect.br().y;

						// Player out of side line (add area condition)
						if (feet > 520 && pCandidates[i]->curRect.area() > 550 && pCandidates[i]->teamID != 2)
						{
							// Initialize ball coordinate at top of player rect
							Point head(pCandidates[i]->curRect.tl().x + pCandidates[i]->curRect.width / 2, pCandidates[i]->curRect.tl().y);

							// Add new candidate and attach it to player head
							BallCandidate* bc = new BallCandidate(curFrame, head, iniRad, 0.0);

							bc->curCrd = head;
							#ifdef WINDOW_PERSPECTIVE
							bc->curRad = perspectiveRad(defRad, bc->curCrd);
							#else
							bc->curRad = defRad;
							#endif	
							bc->fitFrame();
							bc->curAppearM = 10;
							bc->switchState(curFrame, BALL_STATE::TRACKING);

							// Push back curState many times
							bc->updateStep(10); // Dunno

							// Transfer data to bCandidates
							bCandidates.push_back(bc);

							goto done;
						}
					}

					/*******************************************************************
					
												
					
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
						// Performs tracking of all bCandidates and classifies each as Tracking / Searching... Updates lifeTime of Ball [ for removeOutdatedCandidates function ]
						ball_updateBallCandidate(bc, frame, TID);
					}

					
					ball_removeLostCandidates();
					ball_removeStuckedCandidates(0.4);
				done:
					ball_chooseMainCandidate();

					if (mainCandidate != NULL) 
					{

						trackerState = TRACKER_STATE::BALL_FOUND;
					}

					break;
				//_____________________________________________________________
				case TRACKER_STATE::BALL_FOUND : 

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

					break;
				//_____________________________________________________________
				case TRACKER_STATE::TRUE_POSITIVE_FOUND:

					// Task -> truePositives Size can still be 0 when it enters here.. how ?
				
					// if true positive found but candidate gone 
					if (bCandidates.size() < 1 && truePositives.size() > 0)
					{	
						Point3d camCoords;
						Mat H;

						// Camera Coordinates
						for (auto c : truePositives[0]->cameraVisible)
						{
							if (c->id == TID)
							{
								camCoords = c->camCoords;
								H = c->homography;
								break;
							}
						}

						vector<Point2f> p_orig = vector<Point2f>(1);
						vector<Point2f> p_proj = vector<Point2f>(1);

						p_orig[0] = Inv_Triangulate(camCoords, *(truePositives[0]->coords3D.end() - 1));
						perspectiveTransform(p_orig, p_proj, H.inv());

						// A) add a few candidates under a distance constraint
						ball_addMoreCandidates(newCandidates, frame, 2, Point(p_proj[0]/2), TID);
					}
					
					for (auto bc : bCandidates)
					{
						ball_updateBallCandidate(bc, frame, TID);
					}

					// Dunno
					ball_removeLostCandidates();
					ball_removeStuckedCandidates(0.4);
					ball_chooseMainCandidate();

					// Refresh ball candidates if ball not found
					if (mainCandidate == NULL)
					{
						trackerState = TRACKER_STATE::BALL_NOT_FOUND;
						ball_removeOutdatedCandidates(1);
					}
	
					break;		
			}

			if (mainCandidate != NULL) 
			{
				mainCandidateTraj.push_back(mainCandidate->curCrd);
			} 

			else 
			{
				mainCandidateTraj.push_back(outTrajPoint);
			}
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
		void ball_updateBallCandidate(BallCandidate* bc, Mat& frame = Mat(), int TID = 0) {
			
			switch (bc->getState()) {

				//_____________________________________________________________
				case BALL_STATE::TRACKING : {
				//*************************************************************

					vector<double> nearbyP_Dist;
					vector<int> nearbyP_Idx;

					// Find nearest player
					getNearestPlayersVctr(bc, nearbyP_Idx, nearbyP_Dist, 15.0);

					// If player is too near to ball
					if (!nearbyP_Dist.empty()) 
					{
						auto nearest     = min_element(nearbyP_Dist.begin(), nearbyP_Dist.end());
						int  nearestP_Idx = nearbyP_Idx[distance(nearbyP_Dist.begin(), nearest)];

						Rect pRect = pCandidates[nearestP_Idx]->curRect;

						// Determine attach height of ball (bc->attachedHeight)
						updateAttachedHeight(pRect, bc);

						// Update attach height of ball to be at head / mid / feet
						bc->curCrd = Point(pRect.x + pRect.width/2, pRect.y + int(bc->attachedHeight * pRect.height));
						#ifdef WINDOW_PERSPECTIVE
							bc->curRad = Point(pRect.width/2, pRect.height/2) + perspectiveRad(attachRad, bc->curCrd);
						#else
							bc->curRad = Point(pRect.width/2, pRect.height/2) + attachRad;
						#endif // WINDOW_PERSPECTIVE

						bc->fitFrame();
						bc->curAppearM = 0.0;
						bc->switchState(curFrame, BALL_STATE::ATTACHED_TO_PLAYER);
						break;

					}			

					bc->predCrd = bc->curCrd;										// POSITION AT FRAME T-1				
					bc->predCrd = Point(bc->KF.predict().x, bc->KF.predict().y);	// PREDICT FOR FRAME T

					// Correlate
					vector<Point> matchPoints;
					vector<double> matchProbs;
					appearAnalyzer.getMatches(bc, 1, matchPoints, matchProbs);

					// matchPoints is never empty
					if (matchPoints.empty()) 
					{
						bc->switchState(curFrame, BALL_STATE::GOT_LOST);
						break;
					}

					Point nCrd   = matchPoints[0];
					double nProb = matchProbs[0];

					bc->predCrd = bc->KF.correct(nCrd);								// UPDATE FRAME T
					bc->predCrd = Point(bc->KF.predict().x, bc->KF.predict().y);	// PREDICT FOR FRAME T+1
					bc->predCrd = Point(bc->KF.predict().x, bc->KF.predict().y);
					
					// if CC < 0.94
					if (nProb < 0.94) 
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

					// Increase window search area
					#ifdef WINDOW_PERSPECTIVE
						bc->curRad = bc->curRad + perspectiveRad(searchIncRad, bc->curCrd);
					#else
						bc->curRad = bc->curRad + searchIncRad;
					#endif // WINDOW_PERSPECTIVE
						
					bc->fitFrame();

					vector<Point> matchPoints;
					vector<double> matchProbs;

					// Correlate
					appearAnalyzer.getMatches(bc, 1, matchPoints, matchProbs);

					// Evaluate contour if condition is true
					/*float circularity;
					appearAnalyzer.evaluateContours(bc->curCrd, circularity);*/

					Point nCrd   = matchPoints[0];
					double nProb = matchProbs[0];
					/*float nCirc = circularity;*/

					// if CC score > 0.96
					if (nProb > M1_find_threshold /*|| nCirc < 1.4*/) 
					{
						bc->curCrd		= nCrd;
						bc->curAppearM	= nProb;
						/*bc->curCirc		= nCirc;*/

						// set window size to default
						#ifdef WINDOW_PERSPECTIVE
							bc->curRad = perspectiveRad(defRad, bc->curCrd);
						#else
							bc->curRad = defRad;
						#endif // WINDOW_PERSPECTIVE

						bc->fitFrame();

						// switch to BALL_STATE::TRACKING
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

					// Find nearest player
					getNearestPlayersVctr(bc, nearbyP_Idx, nearbyP_Dist);

					if (!nearbyP_Idx.empty()) 
					{
						int minIdx = distance(nearbyP_Dist.begin(), min_element(nearbyP_Dist.begin(), nearbyP_Dist.end()));
						int nearestPlayerIdx = nearbyP_Idx[minIdx];
						double nearestPlayerDist = nearbyP_Dist[minIdx];

						if (nearestPlayerDist < 15.0) 
						{
							// --- move window to the player
							Rect pRect = pCandidates[nearestPlayerIdx]->curRect;
							bc->curCrd = Point(pRect.x + pRect.width/2, pRect.y + int(bc->attachedHeight * pRect.height));

							#ifdef WINDOW_PERSPECTIVE
								bc->curRad = Point(pRect.width/2, pRect.height/2) + perspectiveRad(attachRad, bc->curCrd);
							#else
								bc->curRad = Point(pRect.width/2, pRect.height/2) + attachRad;
							#endif // WINDOW_PERSPECTIVE

							bc->fitFrame();
							bc->curAppearM = 0.0;

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
							appearAnalyzer.getMatches(bc, 3, matchPoints, matchProbs);

							vector<Point>  nCrds = matchPoints;
							vector<double> nProbs = matchProbs;

							// Determine most probable candidate						
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

									// IF - Most probable ball if candidate is isolated
									if (nearestPlayerDist > 50.0 && nProbs[i] > 0.95)
									{
										bc->curCrd = nCrds[i];
										#ifdef WINDOW_PERSPECTIVE
										bc->curRad = perspectiveRad(defRad, bc->curCrd);
										#else
										bc->curRad = defRad;
										#endif // WINDOW_PERSPECTIVE
										bc->fitFrame();
										bc->curAppearM = nProbs[i];
										bc->switchState(curFrame, BALL_STATE::TRACKING);
										break;
									}
								}
							}

							// ELSE - Designate the highest match as the ball
							if (nProbs[0] > M1_find_threshold)
							{
								bc->curCrd = nCrds[0];
								#ifdef WINDOW_PERSPECTIVE
								bc->curRad = perspectiveRad(defRad, bc->curCrd);
								#else
								bc->curRad = defRad;
								#endif // WINDOW_PERSPECTIVE
								bc->fitFrame();
								bc->curAppearM = nProbs[0];
								bc->switchState(curFrame, BALL_STATE::TRACKING);
								break;
							}

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
		void trackPlayers (vector<Rect>& newCandidates, Mat& frame) {
			
			appearAnalyzer.setFrame(frame);

			// calculate coords for all new candidates
			vector<Point> newCandCoords;
			for (auto& c : newCandidates)
			{
				newCandCoords.push_back(Point (c.x + c.width/2, c.y));
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
					
					#ifdef PLAYERS_KF
					p->KF.predict();
					Point cc = p->KF.correct(nCrd);
					p->setCrd(cc, false);
					#else
					p->setCrd(nCrd, false);
					#endif // PLAYERS_KF

					Rect nRect = newCandidates[minIdx];
					p->setRect(nRect);

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
					p->setRect(nCrd);
				}
			}

			// ----- merge players -----
			mergePlayers();

			// ----- add new players -----
			for (unsigned i = 0; i < usedCandidates.size(); i++)
			{
				if (usedCandidates[i] == 0) 
				{
					int pCandidate_teamID = appearAnalyzer.getTeamID(newCandidates[i]);
					pCandidates.push_back(new PlayerCandidate(curFrame, newCandCoords[i], newCandidates[i], pCandidate_teamID));
				}
			}

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
		void getNearestPlayersVctr (BallCandidate* bc, vector<int>& pIndexes, vector<double>& pDistance, double maxDist = numeric_limits<double>::max()) {
			
			// !!! add priority
			pIndexes.clear();
			pDistance.clear();

			vector<Rect> pRects(pCandidates.size());
			vector<double> pDistance_all(pCandidates.size());

			// pRect = container for pCandidate
			for (unsigned i = 0; i < pCandidates.size(); i++) 
			{
				pRects[i] = pCandidates[i]->curRect;
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
		void ball_addMoreCandidates(vector<Point>& possibleCandidates, Mat& frame, int cnt, int TID) {

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
				appearAnalyzer.getMatches(possCand, 1, points, probs);

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

		void ball_addMoreCandidates(vector<Point>& possibleCandidates, Mat& frame, int cnt, Point positionConstraint, int TID) {

			// Create ROI from positionConstraint
			// Evaluate contours
			// Compare output with possibleCandidates
			// Create data -> curCC to complement curAppearM

			vector<BallCandidate*> tCandidates;
			
			// Extract additional candidates through contour analysis
			//vector<pair<float, Point>> newCand = appearAnalyzer.evaluateContours(positionConstraint);

			// Loop through new set of candidates
			for (auto possCand : possibleCandidates)
			{
				bool contains = false;

				// 1) ----- check that new candidate is not inside existent candidates ----- Code will not enter (bCandidates = 0)
				for (auto existCand : bCandidates)
				{
					if (existCand->curRect.contains(possCand))
					{
						contains = true;
						break;
					}
				}

				// 2) ----- check that new candidate is not inside existent players ----- 
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

			// Calculate correlation score for tCandidates
			for (auto possCand : tCandidates)
			{
				vector<Point> points;
				vector<double> probs;

				// Calculate NCC score
				appearAnalyzer.getMatches(possCand, 1, points, probs);

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

			// Calculate circularity score for tCandidates ?

			// From tCandidates, pick the good ones ( can be > 1 ) to be stored into ballCandidates or bc
			sort(tCandidates.begin(), tCandidates.end(), BallCandidate::compareLastAppearM);
			vector<BallCandidate*> toDelete;

			// Keep top 2 candidates. cnt = 2
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
			/*for (auto c : pCandidates) 
			{
				unsigned s = unsigned(max(double(c->coords.size())-50, 1.0));
				for (unsigned i = s; i < c->coords.size(); i++) 
				{
					if		(c->teamID == 1) line(frame, c->coords[i-1], c->coords[i], CV_RGB(0, 0, 255), 1, CV_AA);
					else if (c->teamID == 0) line(frame, c->coords[i-1], c->coords[i], CV_RGB(255, 255, 255), 1, CV_AA);
					else					 line(frame, c->coords[i-1], c->coords[i], CV_RGB(128, 50, 0), 1, CV_AA);
				}
			}*/

			// draw rectangles around the players
			for (auto p : pCandidates) 
			{
				//if (p->ballAttached)
				//{
				//	rectangle(frame, p->curRect, CV_RGB(0,0,255), 1);
				//} 

				//else 
				//{
					if		(p->teamID == 1) rectangle(frame, p->curRect, CV_RGB(0, 0, 255), 1);
					else if (p->teamID == 0) rectangle(frame, p->curRect, CV_RGB(255, 255, 255), 1);
					else if (p->teamID == 2) rectangle(frame, p->curRect, CV_RGB(0, 0, 0), 1);
					else					 rectangle(frame, p->curRect, CV_RGB(128, 50, 0), 1);
				//}
			}

			// ----- display number of players tracked and number of ballCandidates -----
			string label = "";
			//label += to_string(pCandidates.size());

			//if (/*trackerState == TRACKER_STATE::BALL_FOUND || */trackerState == TRACKER_STATE::TRUE_POSITIVE_FOUND)
			//{
			//	label += " YES";
			//} 
			//else if (trackerState == TRACKER_STATE::BALL_NOT_FOUND)
			//{
			//	label += " NO";
			//}
			//else if (trackerState == TRACKER_STATE::BALL_FOUND)
			//{
			//	label += "MAYBE";
			//}
			
			label += /*";  " + */metric.toString_prc();
			setLabel (frame, Rect(0, 0, 900, 20), label);
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
				line(frame, bc->coordsKF[i-1], bc->coordsKF[i], CV_RGB(255, 255, 0), 1, CV_AA);
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
			
			// Height of ball ( but frame t - 1 )
			float height = 0;
			for (int i = 0; i < Ball.size(); i++)	if (TID == Ball[i]->cameraID)	height = (Ball[i]->coords3D.end() - 1)->z;

			//// Track ratio
			double trackRatio = double(bc->getStateDuration(BALL_STATE::TRACKING)) / bc->lifeTime;
			
			// Ball state
			rectangle(frame, bc->curRect, boxColor);
			circle(frame, bc->curCrd, 10, CV_RGB(0, 0, 255));

			// Display text
			char label[40];
			sprintf(label, "%0.2f %d %f", trackRatio, bc->getStateDuration(BALL_STATE::TRACKING), bc->curAppearM/*, bc->id, Ball.size()*//*, bc->curAppearM, Ball.size()*/);
			putText(frame, label, Point((bc->coordsKF.end() - 1)->x + 40, (bc->coordsKF.end() - 1)->y + 10), CV_FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(255, 255, 255));

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
			for (auto i : groups) {
				if (i.empty()) {
					break;
				}
				Rect rectIntersection = (*i.begin())->curRect, rectUnion = (*i.begin())->curRect;
				double mergedProb = 0.0;
				for (auto j : i) {
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
		void mergePlayers() {

			auto groups = merge_(pCandidates, &Tracker::compareMerge_Player);
			
			vector<PlayerCandidate*> toDelete;
			
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

				pCandidates.push_back(new PlayerCandidate(curFrame, mergedCrd, mergedRect, mergedID));
			}

			deleteTrackObjects_(pCandidates, toDelete);
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