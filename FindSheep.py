#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import profile
import datetime
def matchAB(fileA, fileB):
    # 读取图像数据
    imgA = cv2.imread(fileA)
    imgB = cv2.imread(fileB)

    # 取灰度图
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)


    # 获取图片A的大小
    height, width = grayA.shape

    # 取局部图像，寻找匹配位置
    result_window = np.zeros((height, width), dtype=imgA.dtype)
    for start_y in range(0, height-100, 10):
        for start_x in range(0, width-100, 10):
            window = grayA[start_y:start_y+100, start_x:start_x+100]
            match = cv2.matchTemplate(grayB, window, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(match)
            matched_window = grayB[max_loc[1]:max_loc[1]+100, max_loc[0]:max_loc[0]+100]
            result = cv2.absdiff(window, matched_window)
            result_window[start_y:start_y+100, start_x:start_x+100] = result

    # 用四边形圈出不同部分
    _, result_window_bin = cv2.threshold(result_window, 20, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(result_window_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgC = imgA.copy()
    count=0
    for contour in contours:
        min = np.nanmin(contour, 0)
        max = np.nanmax(contour, 0)
        loc1 = (min[0][0], min[0][1])
        loc2 = (max[0][0], max[0][1])
        cv2.rectangle(imgC, loc1, loc2, 255, 2)
        count += 1;

    plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)), plt.title('Original image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)), plt.title('B'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)), plt.title('We Find '+str(count)+" Sheep In The Picture"), plt.xticks([]), plt.yticks([])

    endTimeStamp = datetime.datetime.now()
    print("运行耗时：" + str(endTimeStamp - startTimeStamp) + "秒")
    plt.show()



if __name__ == '__main__':
    startTimeStamp = datetime.datetime.now()


    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_image',
        type=str,
        default='img/image02-00.jpg',
        help='source image'
    )

    parser.add_argument(
        '--target_image',
        type=str,
        default='img/image02-01.jpg',
        help='target image'
    )

    FLAGS, unparsed = parser.parse_known_args()

    #matchAB(FLAGS.source_image, FLAGS.target_image)
    profile.run(matchAB(FLAGS.source_image, FLAGS.target_image))