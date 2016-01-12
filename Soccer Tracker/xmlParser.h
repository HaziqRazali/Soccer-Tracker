#pragma once

#include <cv.h>
#include <vector>
#include "pugixml/src/pugixml.hpp"

using namespace std;
using namespace cv;

namespace st {

//*************************************************************************************************
// ----- This class is used for parsing xml files (e.g. the one with the provided ground truth)
//*************************************************************************************************
class xmlParser {

	//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		xmlParser(void) {}

		//=========================================================================================
		static int parseBallPositions (string fileName, vector<pair<int, Point>>& outVctr) { 

			outVctr.clear();

			pugi::xml_document doc;
			pugi::xml_parse_result result = doc.load_file(fileName.c_str());
			if (!result) {
				return -1;
			}

			pugi::xml_node data = doc.child("viper").child("data").child("sourcefile");
			pugi::xml_node ball = data.find_child_by_attribute("name", "BALL");

			pugi::xml_node ballPos = ball.find_child_by_attribute("name", "BallPos");
			pugi::xml_node ballShot = ball.find_child_by_attribute("name", "BallShot");

			// CLEAN UP
			if ((fileName.compare("ground_truth_ordered\\ground_truth_2.xgtf") == 0) || (fileName.compare("ground_truth_ordered\\ground_truth_4.xgtf") == 0) || (fileName.compare("ground_truth_ordered\\ground_truth_6.xgtf") == 0))
			{
				
				for (pugi::xml_node point : ballPos.children()) {

					string framespan_str = point.attribute("framespan").value();
					string x_str = point.attribute("x").value();
					string y_str = point.attribute("y").value();

					int x = 1920 - stoi(x_str);
					int y = stoi(y_str);

					int frame = stoi(framespan_str.substr(0, framespan_str.find(":")));



					outVctr.push_back(pair<int, Point>(frame, Point(x, y)));
				}
			}

			else
			{
				for (pugi::xml_node point : ballPos.children()) {

					string framespan_str = point.attribute("framespan").value();
					string x_str = point.attribute("x").value();
					string y_str = point.attribute("y").value();

					int x = stoi(x_str);
					int y = stoi(y_str);
					int frame = stoi(framespan_str.substr(0, framespan_str.find(":")));



					outVctr.push_back(pair<int, Point>(frame, Point(x, y)));
				}
			}

			return 0;
		}

		//=========================================================================================
		~xmlParser(void) {}
};

}