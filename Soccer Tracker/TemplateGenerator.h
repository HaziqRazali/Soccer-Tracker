#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;

namespace st {

//*************************************************************************************************
// ----- This class is used to generate images of ball templates
//*************************************************************************************************
class TemplateGenerator {

	//_____________________________________________________________________________________________
	public:

		//=========================================================================================
		TemplateGenerator(void) {}

		////=========================================================================================
		//static Mat makeTemplate () {
		//	Mat templ = imread("c:\\templ4.png");
		//	BALL_DRAW_RAD = Point(templ.cols/2, templ.rows/2);
		//	//BALL_DRAW_RAD = Point(1,1);
		//	//resize(templ, templ, Size(), 0.5, 0.5);
		//	return templ;
		//}

		//=========================================================================================
		static Mat createBallTemple (int rad = 4, int border = 1, Scalar backGrColor = CV_RGB(50,128,50)) {
			int iniScaleCoeff = 10;
			int expRad = rad * iniScaleCoeff;
			int expBorder = border * iniScaleCoeff;
			int templSize = 2 * (expRad + expBorder) + 1;
			Point center(expRad + expBorder, expRad + expBorder);
			Mat templ(templSize, templSize, CV_8UC3, backGrColor);
			circle(templ, center, expRad, CV_RGB(255,255,255), -1);
			int blurRad = rad;
			Size kernelSize(2 * blurRad + 1, 2 * blurRad + 1);
			double sigma = double(blurRad) / 3;
			GaussianBlur(templ, templ, kernelSize, sigma);
			Size finalSize(2 * rad + 1 + border, 2 * rad + 1 + border);
			resize(templ, templ, finalSize, 0, 0, CV_INTER_AREA);
			return templ;
		}

		//=========================================================================================
		static vector<Mat> createBallTemplVctr (	int minRad = 3, int maxRad = 8, int border = 1,
													Scalar backGrColor = CV_RGB(50,100,50)	) {
			vector<Mat> ballTempls;

			for (int i = minRad; i < maxRad; i++) 
			{
				ballTempls.push_back(createBallTemple(i, border, backGrColor));
			}

			/*for (int i = 0; i < ballTempls.size(); i++)
			{
				imshow("haha", ballTempls[i]);
				waitKey(0);
			}*/

			return ballTempls;
		}

		//=========================================================================================
		~TemplateGenerator(void) {}
};

}