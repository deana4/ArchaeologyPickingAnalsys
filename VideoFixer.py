import os
import cv2
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

def changePerspective(videoPath, outputName):
    # PERFECT!!!!!!!!!!!!!!!

    # src_pts = np.float32([[10, 200], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)-1, 220], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)-1, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-1], [0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-1]])
    # dst_pts = np.float32([[0, 100], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.75, 100], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.6, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-100], [100, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-100]])


    # Open the input video file
    cap = cv2.VideoCapture(videoPath)
    # cap = cv2.VideoCapture('dean-hagit.mp4')

    # Define the source and destination points for the projective transformation
    # src_pts = np.float32([[10, 200], [cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 1, 200],
    #                       [cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 1, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 1],
    #                       [0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 1]])
    # dst_pts = np.float32([[0, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.75, 0],
    #                       [cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 200],
    #                       [200, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 200]])
    #
    # src_pts = np.float32([[10, 200], [cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 1, 220],
    #                       [cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 1, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 1],
    #                       [0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 1]])
    # dst_pts = np.float32([[0, 100], [cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 1.2, 100],
    #                       [cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.65, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 100],
    #                       [100, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 100]])
    #
    # src_pts = np.float32([[10, 200], [cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 1, 220],
    #                       [cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 1, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 1],
    #                       [0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 1]])
    # dst_pts = np.float32([[0, 100], [cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.75, 100],
    #                       [cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.6, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 100],
    #                       [100, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 100]])

    # src_pts = np.float32([[100, 70], [cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 1, 0],
    #                       [cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 1, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 1],
    #                       [0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 1]])
    # dst_pts = np.float32([[0, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.75, 0],
    #                       [cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 200],
    #                       [150, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 200]])

    src_pts = np.float32([[10, 200], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)-1, 220], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)-1, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-1], [0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-1]])
    dst_pts = np.float32([[0, 100], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.75, 100], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.6, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-100], [100, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-100]])

    # src_pts = np.float32([[0, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)-1, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)-1, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-1], [0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-1]])
    # dst_pts = np.float32([[0, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.75, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.5, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)], [0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]])

    # src_pts = np.float32([[0, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)-1, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)-1, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-1], [0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-1]])
    # dst_pts = np.float32([[0, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.75, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.5, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)], [0, cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.75]])

    # src_pts = np.float32([[0, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)-1, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)-1, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-1], [0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-1]])
    # dst_pts = np.float32([[30, 100], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.55, 0], [cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.5, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)], [0, cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.75]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Define the codec and create a VideoWriter object to save the output video
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the original video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outputName + ".mp4", fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 1), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 1)))

    # Process each frame of the input video, apply the projective transformation, and write the output to the output video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = cv2.warpPerspective(frame, M, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 1), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 1)))
        out.write(output_frame)

        # cv2.imshow('Output Video', output_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Finished Change Perspective")

def getFiveMinutesOfVid(videoPath, videoFolder, videoName, splittedPath, name = ""):
    # Input video file path
    splitPath = splittedPath + "\\" + videoName.replace(".mp4", "")

    os.makedirs(splitPath, exist_ok=True)

    # Output file path for cropped video
    # output_file = videoPath + "Splitted"

    # Set the start and end times for cropping
    start_time = 0  # Start time in seconds
    end_time = 300  # End time in seconds (5 minutes = 300 seconds)

    # Crop the video using moviepy
    ffmpeg_extract_subclip(videoPath, start_time, end_time, targetname=splitPath + "\\firstFive.mp4")

    video = VideoFileClip(videoPath, verbose=False)

    duration = video.duration

    ffmpeg_extract_subclip(videoPath, duration-300, duration, targetname=splitPath + "\\lastFive.mp4")

    firstFivePath = splitPath + "\\firstFive.mp4"
    lastFivePath = splitPath + "\\firstFive.mp4"

    return firstFivePath, lastFivePath


def getTrayArea(frame, trayX, trayY, segmentX, segmentY):
    frame_copied = frame.copy()
    return frame_copied[trayY : trayY + segmentX, trayX : trayX + segmentY]