import argparse
import cv2

from yolo import YOLO

yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])

def detectHandInFrame(frame):
    width, height, inference_time, results = yolo.inference(frame)

    results.sort(key=lambda x: x[2])

    # how many hands should be shown
    hand_count = len(results)
    # if args.hands != -1:
    #     hand_count = int(args.hands)

    if hand_count > 0:
        return True
    else:
        return False

