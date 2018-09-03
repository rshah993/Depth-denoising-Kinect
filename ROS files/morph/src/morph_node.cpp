#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/rgb_colors.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/opencv.hpp>
//#include </home/shah/ros_catkin_ws/src/morph/include/morph/tracking/include/opencv2/tracking/tracker.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/photo/photo.hpp>
#include <iostream>
#include<vector>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <opencv2/core.hpp>
#include <opencv2/plot.hpp>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

std::vector<float> temp;
std::vector<float> tempr;
std::vector<float> tempg;
std::vector<float> tempb;
std::vector<float> temprgb;
//float data_add[9] = {-0.1,-0.2,-0.3,-0.4,0,0.1,0.20,0.30,0.40};
std::vector<float> nhood(8);
std::vector<float> nhoodr(8);
std::vector<float> nhoodg(8);
std::vector<float> nhoodb(8);
RNG rng;
void remove_zeros(float i)
{
    if(i!=0)
        {
            temp.push_back(i);
        }
}


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
//////    RNG rng(12345);
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }


//    namedWindow("Original",WINDOW_AUTOSIZE);
//    namedWindow("dilated",WINDOW_AUTOSIZE);
//    namedWindow("dilated and eroded",WINDOW_AUTOSIZE);
    Mat &img = cv_ptr->image;
    cvtColor(img,img,CV_BGR2GRAY);
    imshow("Original1",img);
//    cout<<(int) img.at<uchar>(228,138)<<endl;
    Mat dst,kinected,img_rgb,dst_rgb,dst3;
    vector<Mat> channels(3);
    img_rgb = imread("/home/shah/Middlebury/Art/view1.png",CV_LOAD_IMAGE_COLOR);
//    threshold(img, dst, 128, 255, THRESH_BINARY);
//    threshold(img,dilated,128,255, THRESH_BINARY);

    int erosion_size = 3;
    int low;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat element = getStructuringElement(cv::MORPH_CROSS,
                                        cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                        cv::Point(erosion_size, erosion_size) );
//    dst=img.clone();
    Mat original = img.clone();
    Mat mask;
    Canny( img, mask, 30, 30*3, 3 );
    img.setTo(0,mask);
    Canny(img,mask,30, 30*3, 3);
    img.setTo(0,mask);
    Canny( img, mask, 30, 30*3, 3 );
    img.setTo(0,mask);
    Canny(img,mask,30, 30*3, 3);
    img.setTo(0,mask);
    Canny( img, mask, 30, 30*3, 3 );
    img.setTo(0,mask);
    Canny(img,mask,30, 30*3, 3);
    img.setTo(0,mask);

    kinected=img.clone();

    float pa = 0.05;
    float pb = 0;

    int amount1=kinected.rows*kinected.cols*pa;
    int amount2=kinected.rows*kinected.cols*pb;
//    cv::copyMakeBorder( kinected, dst3, 1, 1, 1, 1, cv::BORDER_REFLECT);
//    cout<<"amount1 "<<amount1<<endl;
    for(int counter=0; counter<amount1; ++counter)
    {
         int x = rng.uniform( 0,kinected.rows-2);
         int y = rng.uniform(0, kinected.cols-2);
//         cout<<"x y "<<x<<" "<<y<<endl;
         kinected.at<uchar>(x,y) = 0;
         kinected.at<uchar>(x+1,y) = 0;
         kinected.at<uchar>(x, y+1) = 0;
         kinected.at<uchar>(x+1, y+1) = 0;
         kinected.at<uchar>(x+2, y+1) = 0;
         kinected.at<uchar>(x+1, y+2) = 0;
         kinected.at<uchar>(x+2, y+2) = 0;
         kinected.at<uchar>(x, y+2) = 0;
         kinected.at<uchar>(x+2, y) = 0;

//     cout<<img.at<uchar>(rng.uniform( 0,img.rows), rng.uniform(0, img.cols))<<endl;

    }
//    kinected = dst3.colRange(1, dst3.cols - 2).rowRange(1, dst3.rows - 2).clone();
////    kinected = Mat(dst.rows - 1, dst.cols - 1, CV_8UC1, dst.data);
////    kinected = Mat(kinected.rows - 1, kinected.cols - 1, CV_8UC1, kinected.data + 1*kinected.cols);
//////     for (int counter=0; counter<amount2; ++counter)
//////     {
//////     kinected.at<uchar>(rng.uniform(0,kinected.rows), rng.uniform(0,kinected.cols)) = 255;
//////    }

//////    bitwise_not(img,dst);
////    dilate(kinected,dst,element);
////    erode(dst,dst,element);
//////    dilate(dilated,dilated,element);
    int median;
//    cv::copyMakeBorder( img, dst, 1, 1, 1, 1, cv::BORDER_CONSTANT,0);
//    imshow("Kinect dst",dst);
    Mat dst2 = kinected.clone();
//    for (float check = 10, increment = 10, counter = 1; check <= 1000; check += increment) {
//    cout << "Check = "<<check << endl;
    float check = 100;
    float data_add[9] = {-check,-2*check,-3*check,-4*check,0,check,2*check,3*check,4*check};
    dst=dst2.clone();
    img=kinected.clone();
     imshow("Kinect",img);
    for (int l=1;l<9;l++)
        {
            dst = img.clone();
        for(int i=0; i<img.rows; i++) //now it's right //all starts from zero and ends at n-1. this loop is for 1 to n.
        {
            for(int j=0;j<img.cols;j++)
            {
                if(i==0 && j==0)
                {
                    for(int m=0;m<2;m++)
                    {
                        for(int n=0;n<2;n++)
                        {
                            if(!(m==0 && n==0))
                            {
//                                cout<<"i==0 && j==0"<<m<<n<<endl;
                                nhood.push_back(dst.at<uchar>(i+m,j+n));
                            }
                        }
                    }
                }
//                cout<<"j"<<j<<endl;
                if(i==0 && j<img.cols-1 && j>0)
                {
//                    cout<<i<<j<<endl;
                    for(int m=0;m<2;m++)
                    {
                        for(int n=-1;n<2;n++)
                        {
                            if(!(m==0 && n==0))
                            {
//                                cout<<"i==0 && j<img.cols-1 && j>0"<<endl;
                                nhood.push_back(dst.at<uchar>(i+m,j+n));
                            }
                        }
                    }
                }
                if(i>0 && i<img.rows-1 && j==0)
                {
                    for(int m=-1;m<2;m++)
                    {
                        for(int n=0;n<2;n++)
                        {
                            if(!(m==0 && n==0))
                            {
//                                cout<<"i>0 && i<img.rows-1 && j==0"<<endl;
                                nhood.push_back(dst.at<uchar>(i+m,j+n));
                            }
                        }
                    }
                }
                if(i==img.rows-1 && j<img.cols-1 && j>0)
                {
                    for(int m=-1;m<1;m++)
                    {
                        for(int n=-1;n<2;n++)
                        {
                            if(!(m==0 && n==0))
                            {
//                                cout<<"i==img.rows-1 && j<img.cols-1 && j>0"<<endl;
                                nhood.push_back(dst.at<uchar>(i+m,j+n));
                            }
                        }
                    }
                }
                if(i>0 && i<img.rows-1 && j==img.cols-1 )
                {
                    for(int m=-1;m<2;m++)
                    {
                        for(int n=-1;n<1;n++)
                        {
                            if(!(m==0 && n==0))
                            {
//                                cout<<"i>0 && i<img.rows-1 && j==img.cols-1"<<endl;
                                nhood.push_back(dst.at<uchar>(i+m,j+n));
                            }
                        }
                    }
                }
                if(i==img.rows-1 && j==0)
                {
                    for(int m=-1;m<1;m++)
                    {
                        for(int n=0;n<2;n++)
                        {
                            if(!(m==0 && n==0))
                            {
//                                cout<<"i==img.rows-1 && j==0"<<endl;
                                nhood.push_back(dst.at<uchar>(i+m,j+n));
                            }
                        }
                    }
                }
                if(i==0 && j==img.cols-1)
                {
//                    cout<<"I"<<i<<"J"<<j<<endl;
                    for(int m=0;m<2;m++)
                    {
                        for(int n=-1;n<1;n++)
                        {
                            if(!(m==0 && n==0))
                            {
//                                cout<<"i==0 && j==img.cols-1"<<endl;
                                nhood.push_back(dst.at<uchar>(i+m,j+n));
                            }
                        }
                    }
                }
                if(i==img.rows-1 && j==img.cols-1)
                {
                    for(int m=-1;m<1;m++)
                    {
                        for(int n=-1;n<1;n++)
                        {
                            if(!(m==0 && n==0))
                            {
//                                cout<<"i==img.rows-1 && j==img.cols-1"<<endl;
                                nhood.push_back(dst.at<uchar>(i+m,j+n));
                            }
                        }
                    }
                }
                if(i>0 && i<img.rows-1 && j>0 && j<img.cols-1)
                {
                    for(int m=-1;m<2;m++)
                    {
                        for(int n=-1;n<2;n++)
                        {
                            if(!(m==0 && n==0))
                            {
//                                cout<<"i>0 && i<img.rows-1 && j>0 && j<img.cols-1"<<endl;
                                nhood.push_back(dst.at<uchar>(i+m,j+n));
                            }
                        }
                    }
                }
//                cv::Rect roi(j,i,3,3);
//                nhood=dst(roi).clone().reshape(1,1);
//                std:cout<<typeid(nhood).name()<<std::endl;
                for(int k=0;k<10;k++)
                {
                    data_add[k]=data_add[k]+dst2.at<uchar>(i,j);
//                    cout<<"data_add "<<data_add[k]<<endl;

                }

                    //check size of kyu

                //    float pixelValue = (float)kyu.at<float>(0,k); //In order to get the pixel value of the grayscale
                //image (an integer between 0 and 255), the answer also needs to be typecasted.
                //cv::hconcat(temp,kyu,temp);
//                    for(int k=8;k<16;k++)
//                    {
//                        kyu.at<float>(0,k)=-kyu.at<float>(0,k)+dst.at<uchar>(i,j);
//                    }
                //                for(int p=0;p<nhood.size();p++)
                //                {
                //                   if(nhood[p]!=0)
                //                    {
                for_each (nhood.begin(), nhood.end(), remove_zeros);
//////                    for(int y=0;y<sizeof(nhood);y++)
//////                    {
//////                        if(nhood[y]!=0)
//////                        {
//////                            temp.push_back(nhood[y]);
//////                            tempr.push_back(nhoodr[y]);
//////                            tempg.push_back(nhoodg[y]);
//////                            tempb.push_back(nhoodb[y]);
//////                        }
//////                    }
                //                    break;
                //                    }
                //                }

                if(temp.empty()==1)
                {
                    temp.push_back(0);
//////              tempr.push_back(channels[0].at<uchar>(i,j));
//////              tempg.push_back(channels[1].at<uchar>(i,j));
//////              tempb.push_back(channels[2].at<uchar>(i,j));
                }
//////                    for(int z=0;z<sizeof(temp);z++)
//////                    {
//////
//////                    }
//                    cout<<"after remove zeros"<<endl;
//                    for(int ii=1;ii<temp.size();ii++)
//                    {
//                        std::cout<<temp[ii]<<std::endl;
//                    }
//                    for(int m=0;m<10;m++)
//                    {
//                        temp.push_back(data_add[m]);
//                    }
                //                for_each(kyu.begin(),kyu.end(),add_nhood);

                //                cv::hconcat(temp,kyu,temp);

                //                std::cout<<"in loop part 2 "<<nhood.size()<<std::endl;
                std::sort(temp.begin(),temp.end());
                //                std::cout<<temp.size()<<std::endl;
                //                cv::sort(kyu,kyu,CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
//                cout<<"after sort"<<endl;

                    if (temp.size()%2==0)
                    {
                        median= (int) round((temp[temp.size()/2]+temp[+1+(temp.size()/2)])/2);
                    }
                    else
                    {
                        median=(int) temp[round(temp.size()/2)];
//                        cout<<temp.size()%2<<endl;
                    }
                temp.clear();
                vector<float>(temp).swap(temp);
////                    kyu.assign(zero_vec,zero_vec+9);
//                    cout<<"before data_Add =0;"<<endl;
                data_add[0]=-check;
                data_add[1]=-2*check;
                data_add[2]=-3*check;
                data_add[3]=-4*check;
                data_add[4]=0;
                data_add[5]=1*check;
                data_add[6]=2*check;
                data_add[7]=3*check;
                data_add[8]=4*check;
////                    nhood.assign(zero_vec,zero_vec+8);
////                    cout<<"nhood size "<<sizeof(nhood)<<endl;
                nhood.clear();
                vector<float>(nhood).swap(nhood);
////                int result=take_median(dst,nhood);
////                int median = (int) nhood[13];
////                int pixelValue = (int) median;
                img.at<uchar>(i,j)= median;
//                std::cout<<"median"<< median<<std::endl;


            }
        }
//        cv::copyMakeBorder( img, dst, 1, 1, 1, 1, cv::BORDER_CONSTANT,0);
    }

//////////////////////        img.setTo(0, original<1);
//        imshow("Before new regularizer",img);
////////////////////        dst2 = dst.clone();
////////////////////        for (int l=1;l<6;l++)
////////////////////        {
////////////////////            dst = img.clone();
////////////////////            for(int i=0; i<dst.rows-2; i++) //now it's right //all starts from zero and ends at n-1. this loop is for 1 to n.
////////////////////            for(int j=0;j<dst.cols-2;j++)
////////////////////            {
////////////////////                {
////////////////////                        temp.push_back(dst2.at<uchar>(i,j)+(0.5*10));
////////////////////                        temp.push_back(dst2.at<uchar>(i,j)-(0.5*10));
//////////////////////                        if(dst.at<uchar>(i+1,j))
//////////////////////                        {
////////////////////                            temp.push_back(dst.at<uchar>(i+1,j));
//////////////////////                        }
////////////////////                        std::sort(temp.begin(),temp.end());
////////////////////                        median=(int) temp[round(temp.size()/2)];
////////////////////                        temp.clear();
////////////////////                        vector<float>(temp).swap(temp);
////////////////////                        img.at<uchar>(i,j)= median;
////////////////////
////////////////////                }
////////////////////            }
////////////////////        }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////        img.setTo(0, original<1);
////////        cout<<(int) img.at<uchar>(228,138)<<endl;
//    bitwise_not(dst,dst);
//    findContours( mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
////    cout<<contours.size()<<endl;
//    float radius ;
//    Point2f center ;
//    for(int i=0;i<contours.size();i++)
//    {
//        minEnclosingCircle ( contours[i] , center , radius ) ;
//        cv::circle(img, center, 5, CV_RGB(5, 250, 100), 2);
//    }

//    bitwise_not(mask,mask);
//    bitwise_and(dst,dst,dst,mask);
//    int erosion_size2 = 30;
//    Mat element2 = getStructuringElement(cv::MORPH_CROSS,
//                                        cv::Size(2 * erosion_size2 + 1, 2 * erosion_size2 + 1),
//                                        cv::Point(erosion_size2, erosion_size2) );
//    erode(dilated,dilated,element2);
////////////////////    bitwise_not(img,dst);
////////////////////    threshold(dst,dst,254,255,THRESH_TOZERO);
////////////////////    cv::inpaint(img,dst,dilated,3,cv::INPAINT_TELEA);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    cv::Mat another =img.clone();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////        cv::Mat dilated;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////        bitwise_not(another,dst3);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////        threshold(dst3,dst3,254,255,THRESH_TOZERO);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////        cv::inpaint(another,dst3,dilated,3,cv::INPAINT_TELEA);
    medianBlur(img,img,7);
    dilate(img,img,element);
    imshow("Smooth image", img);
    //imwrite("reindeer_smooth_nocolor.png",img);
//    img=dst.clone();
//    const double C1 = 6.5025, C2 = 58.5225;
//     int d     = CV_32F;
//
//     Mat I1, I2;
//     original.convertTo(I1, d);           // cannot calculate on one byte large values
//     img.convertTo(I2, d);
//
//     Mat I2_2   = I2.mul(I2);        // I2^2
//     Mat I1_2   = I1.mul(I1);        // I1^2
//     Mat I1_I2  = I1.mul(I2);        // I1 * I2
//
//
//
//     Mat mu1, mu2;   //
//     GaussianBlur(I1, mu1, Size(11, 11), 1.5);
//     GaussianBlur(I2, mu2, Size(11, 11), 1.5);
//
//     Mat mu1_2   =   mu1.mul(mu1);
//     Mat mu2_2   =   mu2.mul(mu2);
//     Mat mu1_mu2 =   mu1.mul(mu2);
//
//     Mat sigma1_2, sigma2_2, sigma12;
//
//     GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
//     sigma1_2 -= mu1_2;
//
//     GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
//     sigma2_2 -= mu2_2;
//
//     GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
//     sigma12 -= mu1_mu2;
//
//
//     Mat t1, t2, t3;
//
//     t1 = 2 * mu1_mu2 + C1;
//     t2 = 2 * sigma12 + C2;
//     t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
//
//     t1 = mu1_2 + mu2_2 + C1;
//     t2 = sigma1_2 + sigma2_2 + C2;
//     t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
//
//     Mat ssim_map;
//     divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
//
//     Scalar mssim = mean( ssim_map );
//     cout<<"SSIM "<<mssim[0]<<" PSNR "<<cv::PSNR(original,img)<<endl;
//     imwrite("art_smooth_final.png",img);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////     imwrite("art_kinect_noise.png",kinected);
////////////////////////////////////////////////////////////////////////////////////////////////////////////     Mat xData, yData, display;
////////////////////////////////////////////////////////////////////////////////////////////////////////////     Ptr<cv::plot::Plot2d> plot;
////////////////////////////////////////////////////////////////////////////////////////////////////////////     xData.create(1, 480, CV_64F);//1 Row, 100 columns, Double
////////////////////////////////////////////////////////////////////////////////////////////////////////////     yData.create(1, 480, CV_64F);
////////////////////////////////////////////////////////////////////////////////////////////////////////////     for(int i = 0; i<480; ++i)
////////////////////////////////////////////////////////////////////////////////////////////////////////////    {
////////////////////////////////////////////////////////////////////////////////////////////////////////////        xData.at<double>(i) = i;
////////////////////////////////////////////////////////////////////////////////////////////////////////////        yData.at<double>(i) = img.at<uchar>(200,i) - original.at<uchar>(200,i);
////////////////////////////////////////////////////////////////////////////////////////////////////////////    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////    plot = plot::createPlot2d(xData, yData);
////////////////////////////////////////////////////////////////////////////////////////////////////////////    plot->render(display);
////////////////////////////////////////////////////////////////////////////////////////////////////////////    imshow("Plot", display);
//    imshow("mask",mask);
//    imshow("closing",dilated);
//    imwrite("art_only_median.png",dilated);
//    imshow("dilated and eroded",dilated);
//    imshow("1 time 30",dilated);
    waitKey(30);
//    if (counter == 10) {
//        increment *= 10;
//        counter = 1;
//    }
//    ++counter;
//}
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "morph");
    ros::NodeHandle nh;
    cv::startWindowThread();
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("camera/image", 1, imageCallback);
    ros::spin();
//    ros::MultiThreadedSpinner spinner(16); // Use 4 threads
//    spinner.spin();
}

