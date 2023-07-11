import cv2
import FeatureExtractor

def fixFPS(videoInput, videoOutput, fps):
    cap = cv2.VideoCapture(videoInput)
    l = []

    ret, frame = cap.read()
    col, row, *_ = frame.shape
    while ret:
        l.append(frame)
        ret, frame = cap.read()

    VR = FeatureExtractor.VideoWriter(videoOutput, fps, row, col)
    VR.makeVideo(l)


def fixFPS(videoInput, videoOutput, fps):
    cap = cv2.VideoCapture(videoInput)
    l = []

    ret, frame = cap.read()
    col, row, *_ = frame.shape
    while ret:
        l.append(frame)
        ret, frame = cap.read()

    VR = FeatureExtractor.VideoWriter(videoOutput, fps, row, col)
    VR.makeVideo(l)



if __name__ == "__main__":
    fixFPS("D:\\Python\\Lab_Human_Behavior\\Arc-main\\Splitted\\firstFiveTrayed.mp4", "D:\\Python\\Lab_Human_Behavior\\Arc-main\\Splitted\\downsampled.mp4", fps = 10.0)