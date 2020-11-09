# import the necessary packages
from collections import deque
from imutils.video import VideoStream
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import argparse
import cv2
import imutils
import time
import statistics
import math

#https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
#https://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/
#https://www.pyimagesearch.com/2015/05/04/target-acquired-finding-targets-in-drone-and-quadcopter-video-streams-using-python-and-opencv/
#https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html

# construct the argument parse and parse the arguments
# kill? will be plugging in video

# py opencv_puttcapture_frontview_v01.py --video 200613_Putt10ft_video.mov
# py opencv_puttcapture_frontview_v01.py --video 200613_Putt15ft_video.mov
# py opencv_puttcapture_frontview_v01.py --video 200613_Putt20ft_video.mov
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size")
buffer = 32 #default
args = vars(ap.parse_args())

testing = False #toggle for testing
vidwrite = True #True #toggle to capture video output

# define the lower and upper boundaries of the "green"
# ball in the HSV color space
sensitivity = 30
greenLower = np.array([60-sensitivity,100,80])
greenUpper = np.array([60+sensitivity,255,255])

#area to focus movement detection
xRange = np.array([100,370])
yRange = np.array([250,840])

pts_focusArea = np.array([[0,yRange[1]],[xRange[0],yRange[0]],[xRange[1],yRange[0]],[470,yRange[1]]])

putt_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(0,128,255),(255,255,0),(255,0,128),(128,0,255)]

putt_radius_dist_array = [[1,20],[2,15],[3,10]]
putt_ycoord_dist_array = [[250,280,20],[280,330,15],[330,450,10],[450,600,5],[600,840,3]]

threshold_dist = 80 #define threshold distance to evaluate new object paths
mincntradius = 2 #min radius threshold to filter contours

# colorLower = (29, 86, 6) #green
# colorUpper = (64, 255, 255) #green
# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=buffer)
pts2 = []
puttid = 1
puttlog = []
print('max short-term putt history: ' + str(buffer))
counter = 0
nocnt_counter = 0
(dX, dY) = (0, 0)
direction = ""

if vidwrite:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('PuttDistanceProcessing.avi', fourcc, 10, (640, 480))

#backSub = cv2.createBackgroundSubtractorKNN()
backSub = cv2.createBackgroundSubtractorMOG2(50, 16, False)

def create_region_ofinterest(frame, pts_focusArea):
    ##Make the mask
    #pts_focusArea = pts_focusArea - pts_focusArea.min(axis=0)
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts_focusArea], -1, (255, 255, 255), -1, cv2.LINE_AA)
    ## (3) do bit-op
    dst = cv2.bitwise_and(frame, frame, mask=mask)
    return dst

def create_white_mask(frame):
    sensitivity = 50
    whiteLower = np.array([0,0,255-sensitivity])
    whiteUpper = np.array([180,sensitivity,255])
    blurred = cv2.GaussianBlur(frame, (1, 1), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, whiteLower, whiteUpper)
    mask = cv2.medianBlur(mask,3)
    #mask = cv2.erode(mask, None, iterations=1)
    #mask = cv2.dilate(mask, None, iterations=1)
    return mask

def create_green_mask(frame):
    sensitivity = 30
    greenLower = np.array([60-sensitivity,100,40])
    greenUpper = np.array([60+sensitivity,255,255])
    #blurred = cv2.medianBlur(frame,5)
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.medianBlur(mask,5)
    mask = cv2. bitwise_not(mask)
    #mask2 = cv2.erode(mask2, None, iterations=1)
    #mask = cv2.dilate(mask, None, iterations=1)
    return mask

def create_background_mask(frame, backSub):
    #https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
    #https://www.programcreek.com/python/example/89404/cv2.createBackgroundSubtractorMOG2

    blurred = cv2.GaussianBlur(frame, (1, 1), 0)
    fgMask = backSub.apply(blurred)
    fgMask = cv2.erode(fgMask, None, iterations=1)
    fgMask = cv2.dilate(fgMask, None, iterations=1)
    fgMask = cv2.GaussianBlur(fgMask, (3, 3), 0)
    return fgMask

def get_pts_history(pts, countback):
    x_val = 0
    y_val = 0
    if len(pts) > 0:
        countback = -1 * min(len(pts), countback)
        target_x = [x[0] for x in pts[countback:-1]] #need to fix this
        print(target_x)
        target_y = [x[1] for x in pts[countback:-1]]
        x_val = round(statistics.median(target_x))
        y_val = round(statistics.median(target_y)) #find median of list: https://stackoverflow.com/questions/24101524/finding-median-of-list-in-python
    return (x_val, y_val)

def screen_contours(cnts, pts_history, xRange, yRange):
    new_cnts = []

    #use pts_history as a screen variable (e.g., only use contour similar to historic contour path)

    for cnt in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        #(x, y, w, h) = cv2.boundingRect(cnt)
        #remove if cnt not within detection corridor
        # if y < yRange[0] or y > yRange[1]:
        #     continue
        # if x < xRange[0] or x > xRange[1]:
        #     continue
        #remove if area too big or too small
        if cv2.contourArea(cnt) > 2000 or cv2.contourArea(cnt) < 5:
            #cnts.remove(cnt)
            continue
        #remove if not similar height/width
        # aspectRatio = float(w)/h
        # if aspectRatio < 0.2 or aspectRatio > 4.0:
        #     #cnts.remove(cnt)
        #     continue
        #remove if border is not green? or if low percentage green in rectangle?
        #(x, y, w, h) = cv2.boundingRect(cnt)
        #cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        new_cnts.append(cnt)

    return new_cnts

def draw_contours(frame, cnts):
    for cnt in cnts:
        cv2.drawContours(frame, cnt, -1, (0, 255, 0), 2)
    return frame

def find_top_contour(cnts):
    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and
    # centroid
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) #X, Y - https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
    return center, x, y, radius

#convert putt data to distance estimate
def translate_putt_dist(radius, ycoord):
    # df = pd.DataFrame(putt_radius_dist_array, columns = ['radius', 'distance_ft'])
    # df = df[(df['radius'] == radius)]
    # dist_est = df['distance_ft'].min() #find the distance translation for radius
    #at some point, need to triangulate with angle of descent from ycoord
    #find the distance that corresponds with the ycoord - compare range?
    print('radius: ' + str(radius))
    print('ycoord: ' + str(ycoord))

    df2 = pd.DataFrame(putt_ycoord_dist_array, columns = ['min_ycoord', 'max_ycoord', 'distance_ft'])
    df2 = df2[(df2['min_ycoord'] < ycoord) & (df2['max_ycoord'] >= ycoord)]
    dist_est = df2['distance_ft'].min() #find the distance translation for ycoord

    return dist_est

def write_location_stats(frame, x, y):
    location_text = 'Location (' + str(x) + ', ' + str(y) + ')'
    cv2.putText(frame, location_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def find_direction_and_trace_path(pts, frame, buffer, counter, puttid):
    #print('ptslen: ' + str(len(pts)))
    direction = ''
    dX = 0
    dY = 0
    puttcolor = putt_colors[puttid - 1] #(0, 0, 255)
    frame = write_location_stats(frame, pts[0][0], pts[0][1])
    for i in np.arange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
        if pts[i - 1] is None or pts[i] is None:
            #continue
            raise Exception('no pts in list')
            break
		# check to see if enough points have been accumulated in
		# the buffer
        # if counter >= 10 and i == 1 and pts[-10] is not None:
		# 	# compute the difference between the x and y
		# 	# coordinates and re-initialize the direction
		# 	# text variables
        #     dX = pts[-10][0] - pts[i][0] #should this not be pts[i-10][0]?
        #     dY = pts[-10][1] - pts[i][1]
        #     (dirX, dirY) = ("", "")
		# 	# ensure there is significant movement in the
		# 	# x-direction
        #     if np.abs(dX) > 20:
        #         dirX = "East" if np.sign(dX) == 1 else "West"
		# 	# ensure there is significant movement in the
		# 	# y-direction
        #     if np.abs(dY) > 20:
        #         dirY = "North" if np.sign(dY) == 1 else "South"
		# 	# handle when both directions are non-empty
        #     if dirX != "" and dirY != "":
        #         direction = "{}-{}".format(dirY, dirX)
		# 	# otherwise, only one direction is non-empty
        #     else:
        #         direction = dirX if dirX != "" else dirY

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
        thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], puttcolor, thickness) #draw line between each point

    return direction, frame, dX, dY

def check_path_continuity(center, pts):
    continuous_path = True
    if len(pts) > 0:
        lastcenter = pts[0]
        #print('lastcenter: ' + str(lastcenter))
        #print('currentcenter: ' + str(center))
        dist = math.hypot(center[0] - lastcenter[0], center[1] - lastcenter[1]) #calc distance between points
        #if the distance to last point exceeded threshold break continuity
        if dist > threshold_dist:
            continuous_path = False
    return continuous_path

def update_puttlog(puttlog, puttid):
    putts_array = pts2
    df = pd.DataFrame(putts_array, columns = ['xcoord', 'ycoord', 'puttid', 'radius'])
    df = df[(df['puttid'] == puttid)] #filter records to latest puttid

    cnt_records = len(df.index) #find how many data points for latest puttid
    #if substantial number of records, record as a new putt
    if cnt_records > 10:
        min_ycoord = df['ycoord'].min()
        min_radius = df['radius'].min()
        try:
            dist_est = translate_putt_dist(min_radius, min_ycoord)
            print('distance est: ' + str(dist_est))
        except:
            print('Error: translate putt dist')
        puttlog.append([puttid, int(min_ycoord), int(min_radius), int(dist_est)])

    return puttlog

def write_puttlog(frame, puttlog):
    i = 0
    tot_putts = len(puttlog)
    if tot_putts > 0:
        while i < tot_putts:
            puttlog_text = 'Putt ' + str(puttlog[tot_putts - i - 1][0]) + ': ' + '<Outcome>' + ', ' + str(puttlog[tot_putts - i - 1][3]) + 'ft' #add a distance conversion for radius/angle of descent here
            cv2.putText(frame, puttlog_text, (10, 830 - 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            i += 1

    return frame


#cap = cv2.VideoCapture('200530_Putt2_video.mov')

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
    #vs = cv2.VideoCapture('200613_Putt10ft_video.mov')
# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
	# grab the current frame
    frame = vs.read()
	# handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,then we have reached the end of the video
    if frame is None:
        break
	# rotate frame
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) #rotate clockwise
    frame = cv2.resize(frame[0:1920,0:1080], (472,840))

    if (counter < 2 and testing == True):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

    try:
        mask_white = create_white_mask(frame) #frame
        #cv2.imshow("white mask", mask)

        # mask2 = create_green_mask(frame)
        # cv2.imshow("green mask", mask2) #do i need to invert this mask?

        mask_backsub = create_background_mask(mask_white, backSub)
        cv2.imshow("background mask", mask_backsub)

        mask_roi = create_region_ofinterest(mask_backsub, pts_focusArea)
        cv2.imshow("roi", mask_roi)

        #find intersection of masks?
        #https://html.developreference.com/article/24519203/OpenCV+-+Intersection+between+two+binary+images

    	# find contours in the mask and initialize the current
    	# (x, y) center of the ball
        cnts = cv2.findContours(mask_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        #pts_history = get_pts_history(pts2, 3)
        pts_history = 1
        #print(pts_history)

        cnts = screen_contours(cnts, pts_history, xRange, yRange) #add more filtering to this for a golf ball (below horizon, square-ish aspect ratio, surrounded by green?, has moved xy distance?)
        #print('number of contours: ' + str(len(cnts)))
    	# only proceed if at least one contour was found
        if len(cnts) > 0:
            nocnt_counter = 0 #reset no contour frame counter
            #draw all contours
            frame = draw_contours(frame, cnts)
            #find the largest contour
            center, x, y, radius = find_top_contour(cnts)
    		# only proceed if the radius meets a minimum size

            if radius > mincntradius:

    			# draw the circle and centroid on the frame,
    			# then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                #check if center is significantly significantly far from last point
                continuous_path = check_path_continuity(center, pts)
                if continuous_path == True:
                    pts.appendleft(center)
                    pts2.append([int(x),int(y), puttid, int(radius)])
                else:
                    puttlog = update_puttlog(puttlog, puttid) #update putt log before changing puttid
                    #reset counter, reset pts, reset direction, and iterate puttid
                    pts.clear() #reset pts
                    counter = 0 #reset counter
                    direction = '' #reset direction
                    puttid += 1 #iterate puttid
                    print('puttid: ' + str(puttid))

                    pts.appendleft(center) #add putt to short-term tracker
                    pts2.append([int(x),int(y), puttid, int(radius)]) #add putt ot long-term tracker

                #frame = write_location_stats(frame, int(x), int(y))

        else:
            nocnt_counter += 1 #count how many frames did not have a contour
            #if there have been a few frames without a contour after a previously logged putt, log another putt
            if (nocnt_counter == 10 & puttid > 1):
                puttlog = update_puttlog(puttlog, puttid) #update putt log before changing puttid



        #direction = find_direction(pts)
    	#loop over the set of tracked points

        print('ptslen: ' + str(len(pts)))
        if len(pts) > 0:
            try:
                direction, frame, dX, dY = find_direction_and_trace_path(pts, frame, buffer, counter, puttid)
            except Exception as e:
                print(e)
                continue
    except:
        continue

	# # show the movement deltas and the direction of movement on
	# # the frame
    # cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
    # cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
    #     (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
    #     0.35, (0, 0, 255), 1)

    #write putt log
    if len(puttlog) > 0:
        frame = write_puttlog(frame, puttlog)
    #draw detection corridor on frame
    cv2.drawContours(frame, [pts_focusArea], -1, (0, 255, 0), 2, cv2.LINE_AA)
    #cv2.drawContours(frame, cnt, -1, (0, 255, 0), 2)
    #cv2.rectangle(frame, (xRange[0], yRange[0]), (xRange[1], yRange[1]), (0, 255, 0), 2)

	# show the frame to our screen and increment the frame counter
    cv2.imshow("Frame", frame)
    if vidwrite:
        #frame_out = cv2.resize(frame[250:840,0:840], (640,480))
        frame_out = cv2.resize(frame[250:840,20:820], (640,480))
        out.write(frame_out)
    key = cv2.waitKey(1) & 0xFF
    counter += 1
	# if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    # if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()
# otherwise, release the camera
else:
    vs.release()
    if vidwrite:
        out.release()
# close all windows
cv2.destroyAllWindows()

#print(pts)
print(pts2)
print(puttlog)

#RGB golf ball colors:
#[121, 143, 135] -> [158, 16, 56]
#[96, 118, 102] -> [136, 19, 46]
#[93, 105, 91], [144, 166, 169], [159, 181, 185], [164, 186, 189],
# [213, 227, 208] ->
#, [246, 255, 247]






    # for i in np.arange(1, len(pts)):
	# 	# if either of the tracked points are None, ignore
	# 	# them
    #     if pts[i - 1] is None or pts[i] is None:
    #         continue
	# 	# check to see if enough points have been accumulated in
	# 	# the buffer
    #     if counter >= 10 and i == 1 and pts[-10] is not None:
	# 		# compute the difference between the x and y
	# 		# coordinates and re-initialize the direction
	# 		# text variables
    #         dX = pts[-10][0] - pts[i][0] #should this not be pts[i-10][0]?
    #         dY = pts[-10][1] - pts[i][1]
    #         (dirX, dirY) = ("", "")
	# 		# ensure there is significant movement in the
	# 		# x-direction
    #         if np.abs(dX) > 20:
    #             dirX = "East" if np.sign(dX) == 1 else "West"
	# 		# ensure there is significant movement in the
	# 		# y-direction
    #         if np.abs(dY) > 20:
    #             dirY = "North" if np.sign(dY) == 1 else "South"
	# 		# handle when both directions are non-empty
    #         if dirX != "" and dirY != "":
    #             direction = "{}-{}".format(dirY, dirX)
	# 		# otherwise, only one direction is non-empty
    #         else:
    #             direction = dirX if dirX != "" else dirY
    #
	# 	# otherwise, compute the thickness of the line and
	# 	# draw the connecting lines
    #     thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
    #     cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
