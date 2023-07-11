import math
import cv2
import numpy as np
import threading

class myThread(threading.Thread):
    def __init__(self, target, args =()):
        super().__init__(target=target, args=args)
        self.result = None


    def run(self):
        self.result = self._target(*self._args, **self._kwargs)

class VideoWriter:
    def __init__(self, videoOutPath, fps, row, col):
        self.videoPath = videoOutPath
        output_fps = fps
        output_size = (row, col)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(videoOutPath, fourcc, output_fps, output_size)

    def makeVideo(self, listOfFrames):
        for frame in listOfFrames:
            self.video_writer.write(frame)

        self.video_writer.release()

        return self.videoPath

class Line1:
    def __init__(self):
        self.slope = None
        self.offset = None

    def __call__(self, dot, *args, **kwargs):
        y, x = dot
        val = self.slope * x + self.offset
        if val != y:
            return False

        return True
    # 0,102 - 208, 102 =>
    def calcSlope(self, dot1, dot2):
        y1, x1 = dot1
        y2, x2 = dot2
        if x1 == x2:
            self.slope = 0
        else:
            self.slope = (y2-y1)/(x2-x1)

        return self

    # def calcSlope(self, dot1, dot2):
    #     y1, x1 = dot1
    #     y2, x2 = dot2
    #     if x1 == x2:
    #         self.slope = 0
    #     else:
    #         self.slope = (x2-x1)/(y2-y1)
    #
    #     return self

    #(208, 102)
    def calcOffset(self, dot1):
        self.offset = dot1[1]-self.slope*dot1[0]
        return self

    def changeOffset(self, unit):
        self.offset += unit
        return self

class Line:
    def __init__(self):
        self.slope = None
        self.offset = None
    # 0,0 - 0,102  ---
    def __call__(self, dot, *args, **kwargs):
        x, y = dot
        val = self.slope * x + self.offset
        if val != y:
            return False

        return True

    # dot2 = (0, 102)
    #
    # dot4 = (208, 102)

    def calcSlope(self, dot1, dot2):
        x1, y1 = dot1
        x2, y2 = dot2
        if x1 == x2:
            self.slope = 0
        else:
            self.slope = (y2-y1)/(x2-x1)

        return self

    # def calcSlope(self, dot1, dot2):
    #     y1, x1 = dot1
    #     y2, x2 = dot2
    #     if x1 == x2:
    #         self.slope = 0
    #     else:
    #         self.slope = (x2-x1)/(y2-y1)
    #
    #     return self

    #(0,0)
    def calcOffset(self, dot1):
        self.offset = dot1[1]-self.slope*dot1[0]
        return self

    def changeOffset(self, unit):
        self.offset += unit
        return self

    def printLine(self):
        print("slope: ", self.slope, "offset: ", self.offset)


# listOfFrames = []
#
# def getListOfFrames():
#     return listOfFrames.copy()

# def onLine(dot1, dot2):
#     m = calcSlope(dot1,dot2)
#     b = calcOffset(dot1,m)
#     return *x+calcShift()

def DisplaySplittedTrayIntoCells(frame, trayPointsTup, splitX, splitY):
    if len(trayPointsTup) != 4:
        print("Not enough Points")

    minPoint = min(trayPointsTup)
    maxPoint = max(trayPointsTup)

    startingX = int(minPoint[0])
    startingY = int(minPoint[1])

    endingX = int(maxPoint[0])
    endingY = int(maxPoint[1])

    lastPointX = (startingX, startingY)
    for x in range(startingX, endingX+1, int((endingX - startingX)/splitX)):
        lastPointX = (x, startingY)
        for y in range(startingY, endingY+1, int((endingY - startingY)/splitY)):
            # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.line(frame, lastPointX, (x, y), (0, 0, 255), 1)
            lastPointX = (x, y)

    lastPointY = (startingX, startingY)
    for y in range(startingY, endingY+1, int((endingY - startingY) / splitY)):
        lastPointY = (startingX, y)
        for x in range(startingX, endingX+1, int((endingX - startingX)/splitX)):
            # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.line(frame, lastPointY, (x, y), (0, 0, 255), 1)
            lastPointY = (x, y)

    return endingX - startingX, endingY - startingY, startingX, startingY, frame

def countAmountOfTimesInCell(startingX, startingY, lastPoint, point , splitX, splitY, segmentX, segmentY, countMat):

    # x and y are opposite
    n_point_x = point[0] - startingX
    n_point_y = point[1] - startingY

    mappedX = int(n_point_x / int(segmentY/ splitX))
    mappedY = int(n_point_y / int(segmentX/ splitY))

    # x and y are opposite
    last_n_point_x = lastPoint[0] - startingX
    last_n_point_y = lastPoint[1] - startingY

    last_mappedX = int(last_n_point_x / int(segmentY / splitX))
    last_mappedY = int(last_n_point_y / int(segmentX / splitY))

    if (mappedX >= splitX):
        # print("Skipped")
        return
    elif (mappedY >= splitY):
        # print("Skipped")
        return

    countMat[mappedY, mappedX] += 1

# def keepSandShape(frame):
#     copyFrame = frame.copy()
#     row, col, *_ = frame.shape
#     for x in range(row):
#         for y in range(col):
#             cellVal = copyFrame[x, y]
#             dist = np.linalg.norm(cellVal - (255, 255, 255))
#             if dist > 200:
#                 if np.linalg.norm(cellVal - (255, 0, 0)) < 200:
#                     copyFrame[x, y] = (255, 255, 255)
#                 else:
#                     copyFrame[x, y] = (0, 0, 0)
#             else: #close to white
#                 copyFrame[x, y] = (255, 255, 255)
#
#     return copyFrame

def keepSandShape(frame):
    # Calculate the Euclidean distance between each pixel and (255, 255, 255)
    dist = np.linalg.norm(frame - np.array([255, 255, 255]), axis=2)

    # Create masks for different conditions
    mask1 = dist > 200
    mask2 = np.linalg.norm(frame - np.array([255, 0, 0]), axis=2) < 200

    # Apply the masks to modify the frame
    frame[mask1 & mask2] = np.array([255, 255, 255])
    frame[mask1 & ~mask2] = np.array([0, 0, 0])
    frame[~mask1] = np.array([255, 255, 255])

    return frame

def back_to_visited_cell(point, last_cell, already_visited_mat, splitX, splitY, startingX, startingY,segmentX, segmentY):

    # if point is not in the tray
        #return

    #check in which cell the point is

    #if the point is in cell that is not the cell we were in the previous frame
        #counter++ in this cell of the already_visited_map

    #else, if this is cell that we were in the previous frame we do nothing 

    # in any case, we update the last_cell to be the cell of the point

    n_point_x = point[0] - startingX
    n_point_y = point[1] - startingY

    mappedX = int(n_point_x / int(segmentY/ splitX))
    mappedY = int(n_point_y / int(segmentX/ splitY))

    # the point is not in the tray
    if (mappedX >= splitX):
        return
    elif (mappedY >= splitY):
        return
    
    if (last_cell != (mappedY, mappedX)):
        already_visited_mat[mappedY, mappedX] += 1

    last_cell = (mappedY, mappedX)

    return last_cell

def brush_on_sand_or_paper(frame, point, segmentX, segmentY, startingX, startingY, splitX, splitY):

    #if brush not moving from the previous frame/not in the tray return 0

    #if the brush is on white paper return 1

    #is the brush is on the sand (not white backgroud) return 2
    n_point_x = point[0] - startingX
    n_point_y = point[1] - startingY

    mappedX = int(n_point_x / int(segmentY/ splitX))
    mappedY = int(n_point_y / int(segmentX/ splitY))

    # the point is not in the tray
    if (mappedX >= splitX):
        return 0
    elif (mappedY >= splitY):
        return 0
    
    x = point[0]
    y = point[1]
    cellVal = frame[x, y]

    dist = np.linalg.norm(cellVal - (255, 255, 255))
    
    if dist > 200: #on sand
        return 2
    else: #on paper
        return 1

def brush_on_sand_or_paper_fixed(frame, point, startingX, startingY):
    # if brush not moving from the previous frame/not in the tray return 0

    # if the brush is on white paper return 1

    # is the brush is on the sand (not white backgroud) return 2
    n_point_x = point[0] - startingX
    n_point_y = point[1] - startingY


    row, col, *_ = frame.shape

    # the point is not in the tray
    if (n_point_y >= row) | (n_point_x >= col):
        return "none"


    frame_copy = cv2.circle(frame.copy(), (n_point_x, n_point_y), 1, (0, 255, 0), 2)
    # cv2.imshow('f', frame_copy)
    # cv2.waitKey(1)


    cellVal = frame[n_point_y, n_point_x]
    # print(cellVal)
    # frameToShow = cv2.circle(frame.copy(), (n_point_y, n_point_x), 2, (0, 0, 255), 2)
    #
    # cv2.imshow("frame", frameToShow)
    #
    # if cv2.waitKey(1) == ord('q'):
    #     return
    # dist = numpy.linalg.norm(cellVal - (255, 255, 255))
    B, G, R = cellVal
    if (B, G, R) == (0, 0, 0):  # on sand
        return "sand"
    else:  # on paper
        return "paper"

def movmentOfTheSubjectOnTheGrid(pointOfBrush, splitX, splitY, startingX, startingY, segmentX, segmentY, visitMat, order):
    pointY = pointOfBrush[1] - startingY
    pointX = pointOfBrush[0] - startingX

    mappedX = int(pointX / int(segmentY/ splitX))
    mappedY = int(pointY / int(segmentX/ splitY))

    if (mappedX >= splitX) | (mappedY >= splitY):
        return -1
    elif visitMat[mappedY, mappedX] != 0:
        return -1
    else:
        visitMat[mappedY, mappedX] = order
    return 1

def densityOfSand(blackedFrame, pointOfBrush, startingX, startingY, windowSize):
    pointY = pointOfBrush[1] - startingY
    pointX = pointOfBrush[0] - startingX
    blackedWindow = blackedFrame[pointX - windowSize: pointX + windowSize, pointY - windowSize : pointY + windowSize]

    # cv2.imshow("pic", blackedWindow)
    # cv2.waitKey(0)

    return np.mean(blackedWindow)

def calculateStdOfDustOnTray2(blackedImageOfTray, threshold1=30, threshold2=30, threshold3=29, threshold4=20):
    blackedIM = np.copy(blackedImageOfTray)

    row, col, *_ = blackedImageOfTray.shape

    dot1 = (0, 0)
    dot2 = (0, col-1)

    dot3 = (row-1, 0)
    dot4 = (row-1, col-1)

    #top left->right
    for i in range(row):
        count = 0
        for j in range(col):
            if (dot2[0] - dot1[0]) * (i - dot1[1]) - (j - dot1[0]) * (dot2[1] - dot1[1]) >= 0:
                blackedIM[i, j] = (255, 255, 255)
                if np.all(blackedImageOfTray[i, j] == [0, 0, 0]):
                    count += 1

        if count <= threshold1:
            break

    #left bottom->top
    for i in range(col):
        count = 0
        for j in range(row):
            if (dot3[0] - dot1[0]) * (i - dot1[1]) - (j - dot1[0]) * (dot3[1] - dot1[1]) <= 0:
                blackedIM[j, i] = (255, 255, 255)
                if np.all(blackedImageOfTray[j, i] == [0, 0, 0]):
                    count += 1

        if count <= threshold2:
            break

    #right bottom -> top
    for i in reversed(range(col)):
        count = 0
        for j in range(row):
            if (dot4[0] - dot2[0]) * (i - dot2[1]) - (j - dot2[0]) * (dot4[1] - dot2[1]) <= 0:
                blackedIM[j, i] = (255, 255, 255)
                if np.all(blackedImageOfTray[j, i] == [0, 0, 0]):
                    count += 1

        if count <= threshold3:
            break

    for i in reversed(range(row)):
        count = 0
        for j in range(col):
            if (dot4[0] - dot3[0]) * (i - dot3[1]) - (j - dot3[0]) * (dot4[1] - dot3[1]) >= 0:
                blackedIM[i, j] = (255, 255, 255)
                if np.all(blackedImageOfTray[i, j] == [0, 0, 0]):
                    count += 1

        if count <= threshold4:
            break

    x_tag = 0
    y_tag = 0
    countDots = 0
    for i in range(row):
        for j in range(col):
            if np.all(blackedIM[i, j] == [0, 0, 0]):
                countDots += 1
                x_tag += i
                y_tag += j

    if countDots != 0:
        avgDot = np.array([int(x_tag / countDots), int(y_tag / countDots)])
        avgDotForTheStupidHollandGuy = np.array([int(y_tag / countDots), int(x_tag / countDots)])
    else:
        avgDot = (0, 0)
        avgDotForTheStupidHollandGuy = (0, 0)

    cv2.circle(blackedIM, avgDotForTheStupidHollandGuy, 1, (0, 0, 255), 3)

    cumlativeDist = 0
    for i in range(row):
        for j in range(col):
            r, g, b = blackedIM[i, j]
            if np.all([r, g, b] == [0, 0, 0]):
                cumlativeDist += np.linalg.norm(avgDot - (i, j))

    if cumlativeDist != 0 and countDots > 150:
        cv2.imshow('im1', blackedIM)
        cv2.waitKey(1)

    if countDots != 0:
        if countDots > 150:
            return math.sqrt(cumlativeDist / countDots)
        else:
            return 0
    else:
        return 0

def calculateStdOfDustOnTray(blackedImageOfTray, threshold1 = 30, threshold2 = 30, threshold3 = 29, threshold4 = 20):
    blackedIM = np.copy(blackedImageOfTray)

    row, col, *_ = blackedImageOfTray.shape


    dot1 = (0, 0)
    dot2 = (0, col-1)

    dot3 = (row-1, 0)
    dot4 = (row-1, col-1)


    line1 = Line().calcSlope(dot1, dot2).calcOffset(dot1)
    line2 = Line().calcSlope(dot1, dot3).calcOffset(dot1)
    # line3 = Line().calcSlope(dot2, dot4).calcOffset(dot4)
    line3 = Line().calcSlope(dot2, dot4).calcOffset(dot2)
    line4 = Line().calcSlope(dot3, dot4).calcOffset((col-1, row-1))

    #top left->right
    for i in range(row):
        count = 0
        for j in range(col):
            if line1((j, i)):
                # print("im here")
                blackedIM[i, j] = (255, 255, 255)
                r, g, b = blackedImageOfTray[i, j]
                if [r, g, b] == [0, 0, 0]:
                    count += 1

        if count <= threshold1:
            break
        else:
            line1.changeOffset(1)


    #left bottom->top
    for i in range(col):
        count = 0
        for j in range(row):
            if line2((j, i)):
                blackedIM[j, i] = (255, 255, 255)
                r, g, b = blackedImageOfTray[j, i]
                if [r, g, b] == [0, 0, 0]:
                    count += 1

        if count <= threshold2:
            break
        else:
            line2.changeOffset(1)

    #right bottom -> top
    for i in reversed(range(col)):
        count = 0
        for j in range(row):
            if line3((j, i)):
                blackedIM[j, i] = (255, 255, 255)
                r, g, b = blackedImageOfTray[j, i]
                if [r, g, b] == [0, 0, 0]:
                    count += 1

        if count <= threshold3:
            break
        else:
            line3.changeOffset(-1)


    for i in reversed(range(0, row)):
        count = 0
        for j in range(col):
            if line4((j, i)):
                blackedIM[i, j] = (255, 255, 255)
                r, g, b = blackedImageOfTray[i, j]
                if [r, g, b] == [0, 0, 0]:
                    count += 1

        if count <= threshold4:
            break
        else:
            line4.changeOffset(-1)


    row, col, *_ = blackedImageOfTray.shape
    x_tag = 0
    y_tag = 0
    countDots = 0
    for i in range(row):
        for j in range(col):
            r, g, b = blackedIM[i, j]
            if [r, g, b] == [0, 0, 0]:
                #cv2.circle(blackedIM, (j,i), 1, (0, 0, 255), 3)
                countDots += 1
                x_tag += i
                y_tag += j
    if countDots != 0:
        avgDot = np.array([int(x_tag/countDots), int(y_tag/countDots)])
        avgDotForTheStupidHollandGuy = np.array([int(y_tag / countDots), int(x_tag / countDots)])
    else:
        avgDot = (0,0)
        avgDotForTheStupidHollandGuy = (0,0)



    cv2.circle(blackedIM, avgDotForTheStupidHollandGuy, 1, (0,0,255), 3)


    cumlativeDist = 0
    for i in range(row):
        for j in range(col):
            r, g, b = blackedIM[i,j]
            if [r, g, b] == [0, 0, 0]:
                cumlativeDist += np.linalg.norm(avgDot - (i, j))

    if (cumlativeDist != 0) & (countDots > 150):
        cv2.imshow('im1', blackedIM)
        cv2.waitKey(1)

    if countDots != 0:
        if countDots > 150:
            return math.sqrt(cumlativeDist/countDots)
        else:
            return 0
    else:
        return 0

def calculateStdOfDustOnTray3(blackedImageOfTray, listOfFrames, frame_num, threshold1 = 30, threshold2 = 30, threshold3 = 29, threshold4 = 20):
    blackedIM = np.copy(blackedImageOfTray)


    cv2.imshow('bimg', blackedImageOfTray)
    cv2.waitKey(1)

    row, col, *_ = blackedImageOfTray.shape
    # print("printing rowcol ", row, col)
    # dot1 = (0,0)
    # dot2 = (col-1, 0)
    #
    # dot3 = (0, row-1)
    # dot4 = (col-1, row-1)

    dot1 = (0, 0)
    dot2 = (0, col-1)

    dot3 = (row-1, 0)
    dot4 = (row-1, col-1)

    # dot1 = (0, 0)
    # dot2 = (0, 102)
    #
    # dot3 = (208, 0)
    # dot4 = (208, 102)


    # cv2.line(blackedIM, dot1, dot2, (0,255,0) , 10)
    # cv2.line(blackedIM, dot1, dot3, (0,200,0) , 10)
    # cv2.line(blackedIM, dot2, dot4, (0,150,0) , 10)
    # cv2.line(blackedIM, dot3, dot4, (0,100,0) , 10)
    #
    # cv2.circle(blackedIM, dot1, 1, (0,0,255), 4)
    # cv2.circle(blackedIM, dot2, 1, (0,255,0), 4)
    # cv2.circle(blackedIM, dot3, 1, (255,0,0), 4)
    # cv2.circle(blackedIM, dot4, 1, (244,123,255), 4)

    line1 = Line().calcSlope(dot1, dot2).calcOffset(dot1)
    line2 = Line().calcSlope(dot1, dot3).calcOffset(dot1)
    # line3 = Line().calcSlope(dot2, dot4).calcOffset(dot4)
    line3 = Line().calcSlope(dot2, dot4).calcOffset(dot2)
    line4 = Line().calcSlope(dot3, dot4).calcOffset((col-1, row-1))

    # print(row-1, col-1)
    # line1.printLine()
    # line2.printLine()
    # line3.printLine()
    # line4.printLine()

    #top left->right
    for i in range(row):
        count = 0
        for j in range(col):
            if line1((j, i)):
                # print("im here")
                blackedIM[i, j] = (255, 255, 255)
                r, g, b = blackedImageOfTray[i, j]
                if [r, g, b] == [0, 0, 0]:
                    count += 1

        if count <= threshold1:
            break
        else:
            line1.changeOffset(1)


    #left bottom->top
    for i in range(col):
        count = 0
        for j in range(row):
            if line2((j, i)):
                blackedIM[j, i] = (255, 255, 255)
                r, g, b = blackedImageOfTray[j, i]
                if [r, g, b] == [0, 0, 0]:
                    count += 1

        if count <= threshold2:
            break
        else:
            line2.changeOffset(1)

    #right bottom -> top
    for i in reversed(range(col)):
        count = 0
        for j in range(row):
            if line3((j, i)):
                blackedIM[j, i] = (255, 255, 255)
                r, g, b = blackedImageOfTray[j, i]
                if [r, g, b] == [0, 0, 0]:
                    count += 1

        if count <= threshold3:
            break
        else:
            line3.changeOffset(-1)


    for i in reversed(range(0, row)):
        count = 0
        for j in range(col):
            if line4((j, i)):
                blackedIM[i, j] = (255, 255, 255)
                r, g, b = blackedImageOfTray[i, j]
                if [r, g, b] == [0, 0, 0]:
                    count += 1

        if count <= threshold4:
            break
        else:
            line4.changeOffset(-1)


    row, col, *_ = blackedImageOfTray.shape
    x_tag = 0
    y_tag = 0
    countDots = 0
    for i in range(row):
        for j in range(col):
            r, g, b = blackedIM[i, j]
            if [r, g, b] == [0, 0, 0]:
                #cv2.circle(blackedIM, (j,i), 1, (0, 0, 255), 3)
                countDots += 1
                x_tag += i
                y_tag += j
    if countDots != 0:
        avgDot = np.array([int(x_tag/countDots), int(y_tag/countDots)])
        avgDotForTheStupidHollandGuy = np.array([int(y_tag / countDots), int(x_tag / countDots)])
    else:
        avgDot = (0,0)
        avgDotForTheStupidHollandGuy = (0,0)



    cv2.circle(blackedIM, avgDotForTheStupidHollandGuy, 1, (0,0,255), 3)


    cumlativeDist = 0
    for i in range(row):
        for j in range(col):
            r, g, b = blackedIM[i,j]
            if [r, g, b] == [0, 0, 0]:
                cumlativeDist += np.linalg.norm(avgDot - (i, j))

    # cv2.imshow('avgDot', blackedIM)
    # cv2.waitKey(0)
    #uncomment
    #print(countDots)

    cv2.imshow('im', blackedIM)
    cv2.waitKey(1)

    if (cumlativeDist != 0) & (countDots > 150):
        listOfFrames.append(blackedIM.copy())  # Maybe copy isnt necessary
        # print(frame_num, countDots)


    # cv2.imshow('im', blackedIM)
    # cv2.waitKey(1)

    if countDots != 0:
        if countDots > 150:
            return math.sqrt(cumlativeDist/countDots)
        else:
            return 0
    else:
        return 0
    # uncomment

def findSTDFromBlackVideo(videoInput, videoOutput = "", fps = 30.0, saveVideo = False):
    std = 0
    countFramesWorkedOn = 0
    cap = cv2.VideoCapture(videoInput)
    l = []
    new_l = []
    threadListSTD = []
    ret, frame = cap.read()
    col, row, *_ = frame.shape
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while ret:
        f_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if f_num % 100 == 0:
            print(round((f_num/frame_count)*100, 2), "%")

        t = myThread(target=calculateStdOfDustOnTray, args=([frame, new_l]))
        t.start()
        threadListSTD.append(t)
        ret, frame = cap.read()

    # for thread in threadListSTD:
    #     thread.start()

    for thread in threadListSTD:
        thread.join()

    for thread in threadListSTD:
        retVal = thread.result
        if retVal != 0:
            std += retVal
            countFramesWorkedOn += 1

    if countFramesWorkedOn == 0:
        return 0
    else:
        return (std/countFramesWorkedOn)



    # def findSTDFromBlackVideo(videoInput, videoOutput="", fps=30.0, saveVideo=False):
    #     std = 0
    #     countFramesWorkedOn = 0
    #     cap = cv2.VideoCapture(videoInput)
    #     l = []
    #     new_l = []
    #     ret, frame = cap.read()
    #     col, row, *_ = frame.shape
    #
    #     while ret:
    #         retVal = calculateStdOfDustOnTray(frame, new_l)
    #         if retVal != 0:
    #             std += retVal
    #             countFramesWorkedOn += 1
    #         ret, frame = cap.read()


    # print(len(new_l))
    # if saveVideo:
    #     VW = VideoWriter(videoOutput, fps, row, col)
    #     VW.makeVideo(new_l)
    # if countFramesWorkedOn == 0:
    #     return 0

    # return std/countFramesWorkedOn

def findSTDFromBlackFrames(frames, videoOutput, fps, saveVideo = False):
    std = 0
    countFramesWorkedOn = 0
    new_l = []
    col, row, *_ = frames[0].shape

    for i, frame in enumerate(frames):
        # if i > 200:
        #     break
        retVal = calculateStdOfDustOnTray2(frame, new_l, i)
        if retVal != 0:
            std += retVal
            countFramesWorkedOn += 1


    # print(len(new_l))
    if saveVideo:
        VW = VideoWriter(videoOutput, fps, row, col)
        VW.makeVideo(new_l)

    return std/countFramesWorkedOn

def findSTDFromFrames(frame):
    # Load the image in grayscale
    image = frame
    realImg = image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to separate tray and sand
    # Apply thresholding to separate tray and sand
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations on tray
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


    # Find the largest contour (tray)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tray_contour = None
    largest_area = 0

    # Find the largest contour that represents the tray
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            tray_contour = contour

    # Check if the tray contour is found
    if tray_contour is not None:
        # Create a binary mask for the tray
        tray_mask = np.zeros_like(gray)
        cv2.drawContours(tray_mask, [tray_contour], -1, 255, cv2.FILLED)

        # Apply bitwise AND operation to isolate sand
        sand = cv2.bitwise_and(gray, tray_mask)


        # Compute standard deviation of sand intensity
        std_dev = np.std(sand)
        std_mean = np.mean(sand)
        if std_dev > 7:
            cv2.imshow('stdHigh', sand)
            cv2.waitKey(1)

        print(f"Standard Deviation of sand intensity is {std_dev} and the mean: {std_mean}")
    else:
        print("No tray contour found. Skip this frame.")


    # print(f"Standard Deviation of sand intensity: {std_dev, std_mean}")

    return std_dev

def cut_frame(frame, radius_x, radius_y):
    # Get the frame dimensions
    height, width, *_ = frame.shape

    # Calculate the center coordinates
    center_x = width // 2
    center_y = height // 2

    # Calculate the coordinates for the top-left corner of the cropped region
    top_left_x = max(0, center_x - radius_x)
    top_left_y = max(0, center_y - radius_y)

    # Calculate the coordinates for the bottom-right corner of the cropped region
    bottom_right_x = min(width, center_x + radius_x)
    bottom_right_y = min(height, center_y + radius_y)

    # Crop the frame based on the calculated coordinates
    cropped_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return cropped_frame





















