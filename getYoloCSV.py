import argparse
import configparser
import csv
import os

import cv2
from tqdm import tqdm
from ultralytics import YOLO


def get_points(results, tool, side):
    threshold = 0.5
    if len(results.boxes.data) > 0:
        x1, y1, x2, y2, score, class_id = results.boxes.data[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if score > threshold:
            x_len = abs(x2 - x1)
            y_len = abs(y2 - y1)
            if (y_len > x_len):
                upper_left = (x1, y1)
                upper_right = (x2, y1)
                lower_left = (x1, y2)
                lower_right = (x2, y2)
                trayPoints = [upper_left, upper_right, lower_left, lower_right]
            else:
                return "None"
            if tool == "tray":
                return trayPoints
            elif side == 'L':
                return lower_left
            else:
                return lower_right
    return "None"


def useYolo(video_path, yolo_path, out_path, csv_name, side):
    csv_file_path = os.path.join(out_path, f'{csv_name}.csv')
    if os.path.isfile(csv_file_path):
        print(f"The CSV file '{csv_name}.csv' already exists in the output path.\n")
        return
    csv_file = open(csv_file_path, mode='w', newline='')
    fieldnames = ['frameID', 'trayPoints', 'brushPoint', 'TwizzersPoint']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame_jump = 0

    brush_model = YOLO(f"{yolo_path}/brush/best.pt")
    tray_model = YOLO(f"{yolo_path}/tray/best.pt")
    tweezers_model = YOLO(f"{yolo_path}/tweezers/best.pt")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=frame_count, desc='Processing frames', unit='frame')

    while ret:
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        tray_results = tray_model(frame, verbose=False)[0]
        brush_results = brush_model(frame, verbose=False)[0]
        tweezers_results = tweezers_model(frame, verbose=False)[0]
        trayPoints = get_points(tray_results, "tray", side)
        brushPoint = get_points(brush_results, "brush", side)
        TwizzersPoint = get_points(tweezers_results, "tweezers", side)

        if side == 'L':
            writer.writerow({'frameID': frame_number, 'trayPoints': trayPoints, 'brushPoint': brushPoint,
                             'TwizzersPoint': TwizzersPoint})
        else:
            writer.writerow({'frameID': frame_number, 'trayPoints': trayPoints, 'brushPoint': brushPoint,
                             'TwizzersPoint': TwizzersPoint})
        # frame_jump += 30
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_jump)
        ret, frame = cap.read()
        progress_bar.update(1)

    progress_bar.close()
    csv_file.close()
    cap.release()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid', help='Video folder path')
    parser.add_argument('--yolo', help='Yolo folder path')

    args = parser.parse_args()

    videoPath = args.vid
    yoloPath = args.yolo

    if not videoPath or not yoloPath:
        config = configparser.ConfigParser()
        config.read('featuresConfig.ini')

        videoPath = config.get('Paths', 'video_path')
        yoloPath = config.get('Paths', 'yolo_path')
    directories = [os.path.join(videoPath, name) for name in os.listdir(videoPath) if
                   os.path.isdir(os.path.join(videoPath, name))]

    for dir in directories:
        videos = [file for file in os.listdir(dir) if file.lower().endswith(('.mp4', '.mov'))]

        for video in videos:
            video_name = os.path.splitext(video)[0]
            new_folder_path = os.path.join(dir, video_name + '_YOLO')

            os.makedirs(new_folder_path, exist_ok=True)

            videoPath = os.path.join(dir, video)

            print(f"working on {video}\n"
                  f"-------------------------------------------------------\n")
            useYolo(videoPath, yoloPath, new_folder_path, video_name, "R")
