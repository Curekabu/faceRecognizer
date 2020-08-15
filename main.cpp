#include <stdio.h>
#include <stdlib.h>
#include "arcsoft_face_sdk.h"
#include "amcomdef.h"
#include "asvloffscreen.h"
#include "merror.h"
#include <direct.h>
#include <iostream>  
#include <stdarg.h>
#include <string>
#include "opencv2/opencv.hpp"
#include <windows.h>
#include <time.h>

#pragma comment(lib, "libarcsoft_face_engine.lib")
#pragma comment(lib,"opencv_world411d.lib");

#define SafeFree(p) { if ((p)) free(p); (p) = NULL; }

//����Ϊ�Լ�ע��ArcFace SDK�õ���APPID��SDKKEY 
#define APPID "xxxxxxxxxx"
#define SDKKEY "xxxxxxxxxxx"

using namespace cv;


int main()
{
	MRESULT res = MOK;

	//cv������ͷ
	VideoCapture capture = VideoCapture(0);
	if (capture.isOpened())
		printf("capture open sucessed\n");
	else
		printf("capture open failed\n");


	//��������ӿڣ�ֻ��Ҫ����һ�� 
	//res = ASFOnlineActivation((char*)APPID, (char*)SDKKEY);
	//if (MOK != res && MERR_ASF_ALREADY_ACTIVATED != res)
	//	printf("ASFActivation fail: %d\n", res);
	//else
	//	printf("ASFActivation sucess: %d\n", res);


	//��ʼ���ӿ�
	MHandle handle = NULL;      //�ӿ�handel
	MInt32 mask = ASF_FACE_DETECT | ASF_FACERECOGNITION;
	res = ASFInitEngine(ASF_DETECT_MODE_VIDEO, ASF_OP_0_ONLY, 16, 5, mask, &handle);
	if (res != MOK)
		printf("ASFInitEngine fail: %d\n", res);
	else
		printf("ASFInitEngine sucess: %d\n", res);


	while (1)
	{
		Mat image;
		capture.read(image);
\
		//�������
		ASF_MultiFaceInfo detectedFaces = { 0 };
		res = ASFDetectFaces(handle, image.cols, image.rows, ASVL_PAF_RGB24_B8G8R8, image.data, &detectedFaces);

		if (res == MOK && detectedFaces.faceNum > 0)
		{
			//�ڼ����������ϻ��Ʒ����Լ�ID
			for (auto i = 0; i < detectedFaces.faceNum; ++i)
			{
				rectangle(image, Rect(
					detectedFaces.faceRect[i].left,
					detectedFaces.faceRect[i].top,
					detectedFaces.faceRect[i].right - detectedFaces.faceRect[i].left,
					detectedFaces.faceRect[i].bottom - detectedFaces.faceRect[i].top), Scalar(255, 0, 0));

				putText(image, std::to_string(
					detectedFaces.faceID[i]),
					Point(detectedFaces.faceRect[i].left, detectedFaces.faceRect[i].bottom - 2),
					FONT_HERSHEY_PLAIN,
					1,
					Scalar(255, 0, 0));
			}
		}


		imshow("image", image);
		int key = waitKey(1);


		if (key == 's')
		{
			if (detectedFaces.faceNum == 1)
			{
				//�����ȶ� - �Ǽǵ�һ������
				ASF_FaceFeature feature1 = { 0 };
				ASF_SingleFaceInfo singleDetectedFaces = { 0 };
				singleDetectedFaces.faceRect.left = detectedFaces.faceRect[0].left;
				singleDetectedFaces.faceRect.top = detectedFaces.faceRect[0].top;
				singleDetectedFaces.faceRect.right = detectedFaces.faceRect[0].right;
				singleDetectedFaces.faceRect.bottom = detectedFaces.faceRect[0].bottom;
				singleDetectedFaces.faceOrient = detectedFaces.faceOrient[0];

				res = ASFFaceFeatureExtract(handle, image.cols, image.rows, ASVL_PAF_RGB24_B8G8R8,
					image.data, &singleDetectedFaces, &feature1);
				if (res != MOK)
				{
					printf("�������ʧ�ܣ������Ǽ�ʧ�ܣ�");
					continue;
				}
				else
					printf("�Ǽǳɹ�,����ʶ��׶Ρ�\n");

				//����feature1��copyfearure1
				ASF_FaceFeature copyfeature1 = { 0 };
				copyfeature1.featureSize = feature1.featureSize;
				copyfeature1.feature = (MByte*)malloc(feature1.featureSize);
				memset(copyfeature1.feature, 0, feature1.featureSize);
				memcpy(copyfeature1.feature, feature1.feature, feature1.featureSize);

				while (1)
				{
					capture.read(image);

					//�������
					ASF_MultiFaceInfo detectedFaces2 = { 0 };
					res = ASFDetectFaces(handle, image.cols, image.rows, ASVL_PAF_RGB24_B8G8R8, image.data, &detectedFaces2);
					if (res == MOK && detectedFaces2.faceNum > 0)
					{
						for (auto i = 0; i < detectedFaces2.faceNum; ++i)
						{
							//���ڶ�������������
							ASF_FaceFeature feature2 = { 0 };
							ASF_SingleFaceInfo singleDetectedFaces2 = { 0 };
							singleDetectedFaces2.faceRect.left = detectedFaces2.faceRect[i].left;
							singleDetectedFaces2.faceRect.top = detectedFaces2.faceRect[i].top;
							singleDetectedFaces2.faceRect.right = detectedFaces2.faceRect[i].right;
							singleDetectedFaces2.faceRect.bottom = detectedFaces2.faceRect[i].bottom;
							singleDetectedFaces2.faceOrient = detectedFaces2.faceOrient[i];

							res = ASFFaceFeatureExtract(handle, image.cols, image.rows, ASVL_PAF_RGB24_B8G8R8,
								image.data, &singleDetectedFaces2, &feature2);

							//�����ȶ�
							MFloat confidenceLevel;
							res = ASFFaceFeatureCompare(handle, &copyfeature1, &feature2, &confidenceLevel);
					
							if (confidenceLevel > 0.8)
							{
								rectangle(image, Rect(
									detectedFaces2.faceRect[i].left,
									detectedFaces2.faceRect[i].top,
									detectedFaces2.faceRect[i].right - detectedFaces2.faceRect[i].left,
									detectedFaces2.faceRect[i].bottom - detectedFaces2.faceRect[i].top), Scalar(0, 255, 0));

								putText(image, "SUCCESS:" + std::to_string(
									confidenceLevel),
									Point(detectedFaces2.faceRect[i].left, detectedFaces2.faceRect[i].bottom - 2),
									FONT_HERSHEY_PLAIN,
									1,
									Scalar(0, 255, 0));
							}
							else
							{
								rectangle(image, Rect(
									detectedFaces2.faceRect[i].left,
									detectedFaces2.faceRect[i].top,
									detectedFaces2.faceRect[i].right - detectedFaces2.faceRect[i].left,
									detectedFaces2.faceRect[i].bottom - detectedFaces2.faceRect[i].top), Scalar(0, 0, 255));

								putText(image, "FAIL:" + std::to_string(
									confidenceLevel),
									Point(detectedFaces2.faceRect[i].left, detectedFaces2.faceRect[i].bottom - 2),
									FONT_HERSHEY_PLAIN,
									1,
									Scalar(0, 0, 255));
							}
						}
						
					}


					imshow("image", image);
					int key = waitKey(1);
				 
					if (key == 'q')
						break;
					
				}
				SafeFree(copyfeature1.feature);		//�ͷ��ڴ�
				
			}
			else
				printf("��������������Ϊ1���޷��Ǽǡ�\n");
		}

		if (key == 'q')
			break;
	}
	
	destroyAllWindows();
	capture.release();
    return 0;
    
}
