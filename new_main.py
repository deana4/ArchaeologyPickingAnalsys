import argparse
import configparser
import os
import time

from tqdm import tqdm
import numpy as np
import cv2
import FeatureExtractor
import csv as CSV
import pandas as pd
import ast
import math

def getTrayArea(frame, trayX, trayY, segmentX, segmentY):
    frame_copied = frame.copy()
    return frame_copied[trayY : trayY + segmentX, trayX : trayX + segmentY]

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def getValuesFromCSV(csvFile):
    data_frame = pd.read_csv(csvFile, dtype=str, keep_default_na=False, na_values=None)
    data = data_frame.values.tolist()
    for i in range(len(data)):
        data[i] = [ast.literal_eval(value) for value in data[i]]

    return data

def track(videoPath, csvPath):
    # print(videoPath, csvPath)
    count_brushes_rec = 0
    count_twizzers_rec = 0

    switch = 0
    holding = 0  # 0-start, 1-tweezers, 2-brush
    number_of_frames_holding_brush = 0
    number_of_frames_holding_tw = 0
    avg_brush_length = []

    frameToYolo = getValuesFromCSV(csvPath)
    # print(f'Finished Loading Yolo Frame Analysis.')

    flag = 0
    listOfBlackedShape = []

    splitX = 3
    splitY = 3

    countMatBrush = np.zeros((splitX, splitY))
    jumpMatBrush = np.zeros((splitX * splitY, splitX * splitY))
    alredy_visited_mat_brush = np.zeros((splitX, splitY)) - 1  # so the first visit will bring us to zero. the first visit doesnt count as return visit.


    countMatTwizzers = np.zeros((splitX, splitY))
    jumpMatTwizzers = np.zeros((splitX * splitY, splitX * splitY))
    alredy_visited_mat_twizzers = np.zeros((splitX, splitY)) - 1  # so the first visit will bring us to zero. the first visit doesnt count as return visit.
    # stdOfDustOnFrame = 0

    counter_on_sand_brush = 0
    counter_on_paper_brush = 0
    counter_out_of_tray_or_stable_brush = 0

    counter_on_sand_twizzers = 0
    counter_on_paper_twizzers = 0
    counter_out_of_tray_or_stable_twizzers = 0

    amountOfFramesToDivide = 0

    firstFrame = None

    lastBrushPoint = (0, 0)
    lastTwizzersPoint = (0, 0)
    lastTwoPoint = (0, 0)

    last_cell = (-1, -1)

    listOfBrushingOrderTotally = []
    listOfBrushingOrderEveryXframes = np.zeros((splitX, splitY))
    order = 1

    max_segment = 0
    arr_for_segment = None

    firstFrameFlag = 1

    start_time = time.time()
    cap = cv2.VideoCapture(videoPath)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    firstFrame = None

    progress_bar = tqdm(total=frame_count, desc='Processing frames', unit='frame')

    # print("Working on " + str(frame_count) + " Frames.")

    while ret:
        # frame = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_AREA)
        f_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # if f_num % 100 == 0:
        #     hundFramesFinishedTime = time.time()
        #     print(round((f_num/frame_count)*100, 2), "%")
        #     print("Time Took For 100 Frames: ", round(hundFramesFinishedTime - start_time, 2))
        #     start_time = hundFramesFinishedTime


        frameID, trayPoints, brushPoint, twizzersPoint = frameToYolo[f_num]


        if trayPoints is not None:
            point1, point2, point3, point4 = trayPoints
            # segmentY, segmentX, startingX, startingY, markedFrame
            arr_for_segment = FeatureExtractor.DisplaySplittedTrayIntoCells(frame.copy(), [point1, point2, point3, point4], splitX, splitY)

            # cv2.imshow('Frame', arr_for_segment[-1])
            # cv2.waitKey(1)

            temp_segment_y = arr_for_segment[0]
            temp_segment_x = arr_for_segment[1]

            if temp_segment_y*temp_segment_x >= 0.9*max_segment:
                segmentY, segmentX, startingX, startingY, markedFrame = arr_for_segment
                max_segment = temp_segment_y*temp_segment_x

            max_brush_length = min(euclidean_distance(point1, point2),
                                   euclidean_distance(point1, point3)) // 10  # 1/10 size of tray


        # else:
        #     #keep on reading frames if there isnt a tray
        #     ret, frame = cap.read()
        #     continue


        if firstFrameFlag == 1:
            firstFrame = markedFrame
            firstFrameFlag = 0



        #Features that doesn't relate to twizzers and brush Yolo outputs
        trayAreaFrame = getTrayArea(frame, startingX, startingY, segmentX, segmentY)
        # trayAreaFrameRadiused = FeatureExtractor.cut_frame(trayAreaFrame, 50, 80)
        frameInBlack = FeatureExtractor.keepSandShape(trayAreaFrame.copy())

        # cv2.imshow('Frame', trayAreaFrame)
        # cv2.waitKey(1)
        # print(cframe)


        if brushPoint is not None:
            count_brushes_rec += 1
            FeatureExtractor.countAmountOfTimesInCell(startingX, startingY, lastBrushPoint, brushPoint,
                                                      splitX, splitY, segmentX, segmentY, countMatBrush)
            last_cell = FeatureExtractor.back_to_visited_cell(brushPoint, last_cell, alredy_visited_mat_brush, splitX, splitY,
                                                              startingX, startingY, segmentX, segmentY)

            sand_or_paper = FeatureExtractor.brush_on_sand_or_paper_fixed(frameInBlack, brushPoint, startingX, startingY)


            if sand_or_paper == "paper":
                counter_on_paper_brush += 1
            elif sand_or_paper == "sand":
                counter_on_sand_brush += 1
            elif sand_or_paper == "none":
                counter_out_of_tray_or_stable_brush += 1

            # frame = cv2.circle(frame.copy(), brushPoint, 1, (0, 255, 0), 2)
            # cv2.imshow('dot of brush', frame)
            # cv2.waitKey(1)

            if firstFrame is not None:
                firstFrame = cv2.circle(firstFrame, brushPoint, 1, (0, 255, 0), 2)

            if lastBrushPoint != (0, 0):
                distance = euclidean_distance(lastBrushPoint, brushPoint)
                if distance > 0 and distance < max_brush_length:
                    avg_brush_length.append(distance)

            lastBrushPoint = brushPoint

            number_of_frames_holding_brush += 1

            if holding == 0:
                holding = 2
            elif holding != 2:
                holding = 2
                switch += 1

        # if firstFrame is not None:
        #     cv2.imshow('Frame With Dots', firstFrame)
        #     cv2.waitKey(1)

        if twizzersPoint is not None:
            # print(twizzersPoint)
            count_twizzers_rec += 1
            FeatureExtractor.countAmountOfTimesInCell(startingX, startingY, lastTwizzersPoint, twizzersPoint, splitX, splitY, segmentX, segmentY, countMatTwizzers)
            last_cell = FeatureExtractor.back_to_visited_cell(twizzersPoint, last_cell, alredy_visited_mat_twizzers, splitX, splitY,
                                                              startingX, startingY, segmentX, segmentY)

            sand_or_paper = FeatureExtractor.brush_on_sand_or_paper_fixed(frameInBlack, twizzersPoint, startingX, startingY)


            if sand_or_paper == "paper":
                counter_on_paper_twizzers += 1
            elif sand_or_paper == "sand":
                counter_on_sand_twizzers += 1
            elif sand_or_paper == "none":
                counter_out_of_tray_or_stable_twizzers += 1


            # frame = cv2.circle(frame.copy(), twizzersPoint, 1, (0, 255, 0), 2)
            # cv2.imshow('dot of twizzer', frame)
            # cv2.waitKey(1)

            if firstFrame is not None:
                firstFrame = cv2.circle(firstFrame, twizzersPoint, 1, (0, 0, 255), 2)

            lastTwizzersPoint = twizzersPoint

            number_of_frames_holding_tw += 1
            if holding == 0:
                holding = 1
            elif holding != 1:
                holding = 1
                switch += 1


        if (twizzersPoint is None) & (brushPoint is None):
            pass

        progress_bar.update(1)
        ret, frame = cap.read()
        continue

        # if sand_or_paper != -1:
        #     # print("Its Sand\n", "frame number: " + str(f_num))
        #     ret = FeatureExtractor.movmentOfTheSubjectOnTheGrid(bpoint, splitX, splitY, startingX, startingY,
        #                                                         segmentX, segmentY, listOfBrushingOrderEveryXframes,
        #                                                         order)
        #     # print(listOfBrushingOrderEveryXframes)
        #     if ret != -1:
        #         order += 1

    # print("Calculating Post Frame Analysis")

    for i in range(alredy_visited_mat_brush.shape[0]):
        for j in range(alredy_visited_mat_brush.shape[1]):
            if alredy_visited_mat_brush[i, j] == -1:
                alredy_visited_mat_brush[i, j] = 0

    for i in range(alredy_visited_mat_twizzers.shape[0]):
        for j in range(alredy_visited_mat_twizzers.shape[1]):
            if alredy_visited_mat_twizzers[i, j] == -1:
                alredy_visited_mat_twizzers[i, j] = 0

    stdOfDustOnFrame = 0


    if abs(count_brushes_rec - counter_out_of_tray_or_stable_brush) != 0:
        pbrushingOnSand = (counter_on_sand_brush / abs(count_brushes_rec - counter_out_of_tray_or_stable_brush)) * 100
    else:
        pbrushingOnSand = 0

    if abs(count_brushes_rec - counter_out_of_tray_or_stable_brush) != 0:
        pbrushingOnPaper = (counter_on_paper_brush / abs(count_brushes_rec - counter_out_of_tray_or_stable_brush)) * 100
    else:
        pbrushingOnPaper = 0

    if abs(count_twizzers_rec - counter_out_of_tray_or_stable_twizzers) != 0:
        ptwizzeringOnSand = (counter_on_sand_twizzers / abs(count_twizzers_rec - counter_out_of_tray_or_stable_twizzers)) * 100
    else:
        ptwizzeringOnSand = 0

    if abs(count_twizzers_rec - counter_out_of_tray_or_stable_twizzers) != 0:
        ptwizzeringOnPaper = (counter_on_paper_twizzers / abs(count_twizzers_rec - counter_out_of_tray_or_stable_twizzers)) * 100
    else:
        ptwizzeringOnPaper = 0

    total_holding = number_of_frames_holding_brush + number_of_frames_holding_tw
    frames_brush_per_video = number_of_frames_holding_brush / total_holding * 100
    frames_tw_per_video = number_of_frames_holding_tw / total_holding * 100

    return [countMatBrush, alredy_visited_mat_brush, pbrushingOnSand , pbrushingOnPaper, countMatTwizzers,
            alredy_visited_mat_twizzers, ptwizzeringOnSand, ptwizzeringOnPaper,
            switch, frames_brush_per_video, frames_tw_per_video, np.average(avg_brush_length)]


# countMatBrush, alredy_visited_mat_brush, (counter_on_sand_brush/abs(count_brushes_rec - counter_out_of_tray_or_stable_brush))*100, (counter_on_paper_brush/abs(count_brushes_rec- counter_out_of_tray_or_stable_brush))*100, countMatTwizzers, alredy_visited_mat_twizzers, \
    # (counter_on_sand_twizzers/abs(count_twizzers_rec-counter_out_of_tray_or_stable_twizzers))*100, (counter_on_paper_twizzers/abs(count_twizzers_rec-counter_out_of_tray_or_stable_twizzers))*100

def make_csv_file(data, path):

    row, col = data[0][0][0][0].shape
    headres_countMatBrush = [f'countMatBrush {i},{j}' for i in range(row) for j in range(col)]
    headres_AlreadyVisitedBrush = [f'AlreadyVisitedBrush {i},{j}' for i in range(row) for j in range(col)]
    headres_countMatTwizzers = [f'countMatTwizzers {i},{j}' for i in range(row) for j in range(col)]
    headres_AlreadyVisitedTwizzers = [f'AlreadyVisitedTwizzers {i},{j}' for i in range(row) for j in range(col)]
    headres_countMatBrush_last = [f'countMatBrushLast {i},{j}' for i in range(row) for j in range(col)]
    headres_AlreadyVisitedBrush_last = [f'AlreadyVisitedBrushLast{i},{j}' for i in range(row) for j in range(col)]
    headres_countMatTwizzers_last = [f'countMatTwizzersLast {i},{j}' for i in range(row) for j in range(col)]
    headres_AlreadyVisitedTwizzers_last = [f'AlreadyVisitedTwizzersLast {i},{j}' for i in range(row) for j in range(col)]


    fieldnamesFirst = headres_countMatBrush + headres_AlreadyVisitedBrush + ['%OfSandBrushes'] + ['%OfPaperBrushes'] + headres_countMatTwizzers + headres_AlreadyVisitedTwizzers + ['%OfSandTwizzers'] + ['%OfPaperTwizzers'] + ['switch'] + ['frames_brush_per_video'] + ['frames_tw_per_video'] + ['avg_brush_length']
    fieldnamesLast = headres_countMatBrush_last + headres_AlreadyVisitedBrush_last + ['%OfSandBrushesLast'] +  ['%OfPaperBrushesLast'] + headres_countMatTwizzers_last + headres_AlreadyVisitedTwizzers_last + ['%OfSandTwizzersLast'] + ['%OfPaperTwizzersLast'] + ['switchLast'] + ['frames_brush_per_video_last'] + ['frames_tw_per_video_last'] + ['avg_brush_length_last']
    extra_data = ['Tested', 'Half an hour', 'Directory']
    # fieldnamesLast = ['countMatBrush', 'avgSTDLast', 'alreadyVisitMatLast', 'ofBrushesOnSandLast', 'ofBrushesOnPaperLast', 'listOfBrushingOrderEveryXFramesLast', 'jumpMatLast', 'subject', 'videoName']
    if os.path.exists(path):
        val = input("Do you want to re-write the csv? ")
        if val == "no":
            return 0

    csv_file = open(path + "\\features.csv", mode='w', newline='')
    writer = CSV.DictWriter(csv_file, fieldnames=fieldnamesFirst + fieldnamesLast + extra_data)
    writer.writeheader()

    for dataLine in data:
        try:
            firstData, lastData = dataLine[0]
            details_list = []
            dir_name = dataLine[1]
            tested = dir_name.split('-', 1)[0]
            half_hour = dir_name.rsplit('-',1)[1]
            details_list.append(tested)
            details_list.append(half_hour)
            details_list.append(dir_name)
            first_data_new = []
            last_data_new = []
            for i in range(len(firstData)):
                if isinstance(firstData[i], np.ndarray):
                    firstData[i] = firstData[i].flatten().tolist()
                if isinstance(lastData[i], np.ndarray):
                    lastData[i] = lastData[i].flatten().tolist()

            for i in range(len(firstData)):
                for item in firstData:
                    if isinstance(item, list) == False:
                        first_data_new.append(item)
                    else:
                        for val in item:
                            first_data_new.append(val)

            
            for i in range(len(lastData)):
                for item in lastData:
                    if isinstance(item, list) == False:
                        last_data_new.append(item)
                    else:
                        for val in item:
                            last_data_new.append(val)

            #print(first_data_new)
            #last_data_new = [val for sublist in lastData for val in sublist if isinstance(val, float)]

            data_dict = {
                **{key: value for key, value in zip(fieldnamesFirst, first_data_new)},
                **{key: value for key, value in zip(fieldnamesLast, last_data_new)},
                **{key: value for key, value in zip(extra_data, details_list)}
            }
            writer.writerow(data_dict)
        except:
            print("Could not load data correctly")
            continue

    csv_file.close()

# if __name__ == '__main__':
#     val = track("D:\\Python\\Lab_Human_Behavior\\Arc-main\\videos\\1\\firstFive_nir.mp4", "D:\\Python\\Lab_Human_Behavior\\Arc-main\\csvs\\1\\firstFive_nir.csv")

if __name__ == '__main__':
    allVideoFeatures = []

    parser = argparse.ArgumentParser()

    config = configparser.ConfigParser()
    config.read('featuresConfig.ini')

    videoPath = config.get('Paths', 'video_path')
    csvPath = config.get('Paths', 'csv_path')
    csv_final_dir = config.get('Paths', 'csv_features_path')

    videos_directories = [os.path.join(videoPath, name) for name in os.listdir(videoPath) if os.path.isdir(os.path.join(videoPath, name))]
    csv_directories = [os.path.join(csvPath, name) for name in os.listdir(csvPath) if os.path.isdir(os.path.join(csvPath, name))]



    for video_dir, csv_dir in zip(videos_directories, csv_directories):
        totalFeatures = []

        videos = [file for file in os.listdir(video_dir) if file.lower().endswith(('.mp4', '.mov'))]
        csvs = [file for file in os.listdir(csv_dir) if file.lower().endswith('.csv')]

        video_directory_name = os.path.basename(video_dir)
        csv_directory_name = os.path.basename(csv_dir)

        if video_directory_name != csv_directory_name:
            print(f"video directory didn't match csv directory -> skipping directories: video: {video_directory_name}, csv: {csv_directory_name}")
            continue
        # new_folder_path = os.path.join(video_dir, video_files_name + '_feautures')
        # os.makedirs(new_folder_path, exist_ok=True)
        print(f"STARTED working on directory named: {video_directory_name} \n")
        for video, csv in zip(videos, csvs):
            videoName = os.path.basename(video).replace(".mp4", "").replace(".mov", "")
            csvName = os.path.basename(csv).replace(".csv", "")

            if videoName != csvName:
                print(f"video didn't match csv -> skipping directories: video: {videoName}, csv: {csvName}")
                continue

            video_name = os.path.splitext(video)[0]
            csv_name = os.path.splitext(csv)[0]

            videoPath = os.path.join(video_dir, video)
            csvPath = os.path.join(csv_dir, csv)
            print(f"STARTED working on {video} and {csv}\n")
            totalFeatures.append(track(videoPath, csvPath))
            print(f"-------------------------------------------------------")
        allVideoFeatures.append([totalFeatures, video_directory_name])


    make_csv_file(allVideoFeatures, csv_final_dir)

    print("Finished working on all videos.")

