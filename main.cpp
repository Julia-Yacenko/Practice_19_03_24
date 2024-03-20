#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
RNG rng(12345);

int main()
{
    cv::Mat colorImg = cv::imread("D:/Camera/lab.png");
    if (!colorImg.data)
    {
        printf("Error loading image \n"); return -1; 
    }
    cv::imshow("Color Input", colorImg);

    cv::Mat image1;
    GaussianBlur(colorImg, image1, Size(3, 3), 0);
    cv::Mat edges1;
    Canny(image1, edges1, 50, 200);
    cv::imshow("Color Canny", edges1);

    cv::Mat greyImg;
    cv::cvtColor(colorImg, greyImg, COLOR_BGR2GRAY);
    cv::imshow("Grey Input", greyImg);

    cv::Mat image2;
    GaussianBlur(greyImg, image2, Size(5, 5), 0);
    cv::Mat edges2;
    Canny(image2, edges2, 50, 200);
    cv::imshow("Grey Canny", edges2);

    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;

    findContours(edges1, contours, hierarchy, RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE);
    vector<Moments> mu(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i], false);
    }
    vector<Point2f> mc(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }
    vector<vector<Point>> contourPoly(contours.size()); 
    for (size_t i = 0; i < contours.size(); i++)
    {
        int c_area = contourArea(contours[i]); 
        float peri = arcLength(contours[i], true);
        approxPolyDP(contours[i], contourPoly[i], 0.02 * peri, true);
        drawContours(colorImg, contours, (int)i, Scalar(0, 255, 0), 2, LINE_8, hierarchy, 0);

        int obj_corners = (int)contourPoly[i].size();
        string obj_name;

        if (obj_corners == 3)
            obj_name = "Triangle";
        else if (obj_corners == 4) 
        {
            obj_name = "Rectangle";
        }
        else if (obj_corners > 6)
            obj_name = "Circle";

        putText(colorImg, obj_name, 
            mc[i], 
            FONT_HERSHEY_PLAIN, 
            1,
            Scalar(255, 255, 255),
            2);
    } 
    imshow("Color Contours", colorImg);

    /*findContours(edges2, contours, hierarchy, RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(colorImg, contours, (int)i, Scalar(0, 255, 0), 2, LINE_8, hierarchy, 0);
    }
    imshow("Grey Contours", colorImg);*/
    

    cv::Mat textImg = cv::imread("D:/Camera/text.jpg");
    if (!textImg.data)
    {
        printf("Error loading image \n"); return -1;
    }
    //cv::imshow("Text Input", textImg);

    cv::Mat outImg;
    resize(textImg, outImg, Size(500, 500), INTER_LINEAR);
    cv::imshow("Resize", outImg);

    cv::Mat blurImg;
    GaussianBlur(outImg, blurImg, Size(3, 3), 0);
    cv::Mat cannyImg;
    Canny(blurImg, cannyImg, 50, 200);
    cv::imshow("Text Canny", cannyImg);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}