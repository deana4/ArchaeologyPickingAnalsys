import contextlib
import csv
import time

from ultralytics import YOLO
import cv2
import os
import sys


#'tray', 'brush', 'tweezers'

# video_path = 'C:\\Users\\liran\\Hagit Lab\\code\\Arc-main\\video_test_short_black.mp4'
# video_path_out = '{}_brush.mp4'.format(video_path)

# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()
# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Load a model
# model_tweezers = YOLO("D:\\Python\\Lab_Human_Behavior\\Arc-main\\tweezers\\best.pt")
# model_tray = YOLO("D:\\Python\\Lab_Human_Behavior\\Arc-main\\tray\\best.pt")
# model_brush = YOLO("D:\\Python\\Lab_Human_Behavior\\Arc-main\\brush\\best.pt")

def getTrayArea(frame, trayX, trayY, segmentX, segmentY):
    frame_copied = frame.copy()
    return frame_copied[trayY : trayY + segmentX, trayX : trayX + segmentY]

def getTrayPoints(frame):
    isFoundTray, coordinateList = getCoordinates(frame, "tray")
    if isFoundTray == "notFound":
        return isFoundTray, coordinateList
    else:
        x1, y1, x2, y2 = coordinateList
        return isFoundTray, [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]

def getBrushPoints(frame):
    isFoundBrush, coordinateList = getCoordinates(frame, "brush")
    point = (0, 0)
    if isFoundBrush == "notFound":
        return isFoundBrush, point
    else:
        x1, y1, x2, y2 = coordinateList
        point2 = (x2, y2)
        point1 = (x1, y1)

        x_len = abs(x2 - x1)
        y_len = abs(y2 - y1)
        if (x_len > y_len):
            return "notFound", point

        if point1 > point2:
            point = point1
        else:
            point = point2

        return isFoundBrush, point

def getTwizzerPoints(frame):
    isFoundTwizzer, coordinateList = getCoordinates(frame, "tweezers")
    point = (0, 0)
    if isFoundTwizzer == "notFound":
        return isFoundTwizzer, point
    else:
        x1, y1, x2, y2 = coordinateList
        point2 = (x2, y2)
        point1 = (x1, y1)

        x_len = abs(x2 - x1)
        y_len = abs(y2 - y1)
        if (x_len > y_len):
            return "notFound", point

        if point1 > point2:
            point = point1
        else:
            point = point2

        return isFoundTwizzer, point

def getYoloPointsForEachFrameConcurrent(videoPath, csv_name, numOfThreads = 2):
    frameToRecognition = {} # framenumber : [brushPoint, TwizzersPoint]
    cap = cv2.VideoCapture(videoPath)
    ret, frame = cap.read()
    flag = 1
    trayPoints = None
    threadList = []
    startTime = time.time()
    while ret:
        f_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        endTime = time.time()
        if f_num % 100 == 0:
            print(f"time for 100 frames: {endTime - startTime}")
            startTime = time.time()

        # if f_num > 50:
        #     break

        if f_num % numOfThreads == 0:
            for thread in threadList:
                thread.start()

            for thread in threadList:
                thread.join()

            threadList.clear()

        threadList.append(myThread(target=frameWorker, args=([frame, f_num, frameToRecognition])))
        ret, frame = cap.read()

    for thread in threadList:
        thread.start()

    for thread in threadList:
        thread.join()

    threadList.clear()

    print("waiting for the last thread")

    csvName = saveCSVFromDict(frameToRecognition, csv_name)

    return frameToRecognition, csvName

def getYoloPointsForEachFrameSerial(videoPath):
    frameToRecognition = {}  # framenumber : [brushPoint, TwizzersPoint]
    cap = cv2.VideoCapture(videoPath)
    ret, frame = cap.read()
    startTime = time.time()
    while ret:
        f_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        endTime = time.time()
        if f_num % 100 == 0:
            print(f"time for 100 frames: {endTime - startTime}")
            startTime = time.time()
        if f_num > 300:
            break
        frameWorker(frame, f_num, frameToRecognition)
        ret, frame = cap.read()

    return frameToRecognition

def frameWorker(frame, f_num, frameToRecognition):
    print(f_num)

    isFoundTray, trayPoints = getTrayPoints(frame)
    isFoundBrush, brushPoint = getBrushPoints(frame)
    isFoundTwizzers, twizzersPoint = getTwizzerPoints(frame)
    # isFoundBrush, brushPoint = ["notFound", (0,0)]
    # isFoundTwizzers, twizzersPoint = ["notFound", (0,0)]

    tpoint = "None"
    bpoint = "None"
    trayP = "None"

    if isFoundBrush == "found":
        bpoint = brushPoint
    if isFoundTwizzers == "found":
        tpoint = twizzersPoint
    if isFoundTray == "found":
        trayP = trayPoints

    frameToRecognition[f_num] = [trayP, bpoint, tpoint]

def saveCSVFromDict(dictOfPoints, csvName = "videoCSV.csv", csv_columns = ['frameID', 'trayPoints', 'brushPoint', 'TwizzersPoint']):
    try:
        csv_file = open(csvName, mode='w', newline='')
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()

        for i in range(1, len(dictOfPoints)+1):
            # print(dictOfPoints[i], len(dictOfPoints[i]))
            trayPoints, brushPoint, twizzersPoint = dictOfPoints[i]
            writer.writerow({'frameID' : i, 'trayPoints' : trayPoints, 'brushPoint' : brushPoint, 'TwizzersPoint' : twizzersPoint})
    except IOError:
        print("I/O error")

    return csvName

def getCoordinates(frame, tool):
    threshold = 0.5
    model = None
    if tool == "brush":
        model = YOLO("D:\\Python\\Lab_Human_Behavior\\Arc-main\\brush\\best.pt")
    elif tool == "tray":
        model = YOLO("D:\\Python\\Lab_Human_Behavior\\Arc-main\\tray\\best.pt")
    elif tool == "tweezers":
        model = YOLO("D:\\Python\\Lab_Human_Behavior\\Arc-main\\tweezers\\best.pt")
    else:
        print("No Correct Model Founded")
        return -1

    results = model(frame, verbose=False)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            return "found", [int(x1), int(y1), int(x2), int(y2)]

    return "notFound", [int(-1), int(-1), int(-1), int(-1)]



