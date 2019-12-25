
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

const int max_value_H = 360/2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Live2";
int low_H = 0, low_S = 0, low_V = 254;
int high_H = 10, high_S = 100, high_V = 255;

static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H-1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H+1);
    setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S-1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = max(high_S, low_S+1);
    setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V-1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = max(high_V, low_V+1);
    setTrackbarPos("High V", window_detection_name, high_V);
}

int main(int, char**)
{
    Mat frame;
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 0;             // 0 = open default camera
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API
    // open selected camera using selected API
    cap.open(deviceID + apiID);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

	
	namedWindow(window_capture_name);
    namedWindow(window_detection_name);

	  createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);

    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;

int dilation_size = 10;
int erosion_size = 1;
		 Mat elementDil = getStructuringElement( cv::MORPH_ELLIPSE,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
				 Mat elementErode = getStructuringElement( cv::MORPH_ELLIPSE,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );							   
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            continue;
        }

		Mat framecp = frame.clone();

		// Convert input image to HSV
		cv::Mat hsv_image;
		cv::cvtColor(framecp, hsv_image, cv::COLOR_BGR2HSV);

		// Threshold the HSV image, keep only the red pixels
		cv::Mat red_hue_image, blurred;
		cv::inRange(hsv_image, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), red_hue_image);
//		cv::inRange(hsv_image, Scalar(0, 0, 254), Scalar(10, 100, 255), red_hue_image);

		blurred = red_hue_image.clone();
		//cv::medianBlur ( red_hue_image, blurred, 5 );
//		cv::GaussianBlur( red_hue_image, blurred, Size( 5, 5), 0, 0 );
		//cv::GaussianBlur(red_hue_image, red_hue_image, cv::Size(9, 9), 2, 2);

		cv::dilate(blurred, blurred, elementDil);
		cv::erode(blurred, blurred, elementErode);

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		cv::findContours(blurred, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		// get the moments
		vector<Moments> mu(contours.size());
		for( int i = 0; i<contours.size(); i++ )
		{ mu[i] = moments( contours[i], false ); }
		
		// get the centroid of figures.
		vector<Point2f> mc(contours.size());
		for( int i = 0; i<contours.size(); i++)
		{ mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }
		
		// draw contours
		for( int i = 0; i<contours.size(); i++ )
		{
			cv::circle( framecp, mc[i], 10, cv::Scalar(0, 255, 0), 5 );
		}
		
		cout << "Countours : " << contours.size() << "\n";
	    imshow(window_capture_name, framecp);
	    imshow(window_detection_name, blurred);
        if (waitKey(5) >= 0)
            break;
		cout << "Low HSV:  " << low_H << "," << low_S << "," << low_V << " High HSV: " << high_H << "," << high_S << "," << high_V << "\n";
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}