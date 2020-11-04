import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from icp import icp
import sys

class Stereo:
    
    img_L = []
    img_R = []
    kp_l = []
    kp_r = []
    des_l = []
    des_r = []
    points3D = []
    
    def __init__(self, frameNo):
        self.frameNo = frameNo
    
    def Matching(self, img_L, img_R):
        self.img_L = img_L
        self.img_R = img_R
        self.kp_l, self.des_l = self.KpDes(self.img_L)
        self.kp_r, self.des_r = self.KpDes(self.img_R)
        
    def KpDes(self, IMG):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # Initiate FAST detector
        fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
        # find and draw the keypoints
        kp = fast.detect(IMG, None)
        
        # find the keypoints and compute the descriptors with ORB
        kp, des = orb.compute(IMG, kp)
        
        return kp, des
    

def main():
    
    try:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fname = "DepthImage.avi"
        fps = 15.0
        videoWriter = cv2.VideoWriter(fname, fourcc, fps, (1241, 376))
    except:
        print("Error: can't create output video: %s" % fname)
        sys.exit()
    
    maxMatches = 500
    frameNo = 0
    
    currentT_L = np.eye(4)
    currentT_R = np.eye(4)
    
    odom = np.zeros((1024,1024,3))
    kpR = []
    datasetDir = "dataset/sequences/"
    
    sequence = "00"   
    calibFile = open(datasetDir + sequence + "/calib.txt","r")
    timeFile = open("dataset/poses/" + sequence + ".txt","r")
    projectionMatrices = calibFile.read().split()
    
    focalLength = float(projectionMatrices[1])
    baseLength =  -1*float(projectionMatrices[17])/float(projectionMatrices[14]) # base = -P1(1,4)/P1(1,1) (in meters)
    
    K = [[float(projectionMatrices[1]), float(projectionMatrices[2]), float(projectionMatrices[3])],
                       [float(projectionMatrices[5]), float(projectionMatrices[6]), float(projectionMatrices[7])],
                       [float(projectionMatrices[9]), float(projectionMatrices[10]), float(projectionMatrices[11])]]
    
    cx = float(projectionMatrices[3])
    cy = float(projectionMatrices[7])
    t = timeFile.readlines()
    #print(t)
    
    frameNo = 0
    imageNames = os.listdir(datasetDir + sequence + "/image_0")
    imageNames.sort()
    for image in imageNames:
        
        current = Stereo(frameNo)
        
        imgL = cv2.imread(datasetDir + sequence + "/image_0/" + image)
        imgR = cv2.imread(datasetDir + sequence + "/image_1/" + image)
        
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        current.Matching(imgL, imgR)
        
        scale, x, y, z = getAbsoluteScale(frameNo, t)
        print("scale = ", scale)
        
        if frameNo:
            currentT_L, currentT_R, kpL, matchesL, matchesR = poseEstimation(previous, current, focalLength, baseLength , cx , cy, currentT_L, currentT_R, K, scale)
            
        print(currentT_L[:3,3])
        
        odom = cv2.circle(odom, (int(currentT_L[0,3]) + 512, int(currentT_L[2,3]) + 512), 2, (255,0,0), 2)
        odom = cv2.circle(odom, (int(currentT_R[0,3]) + 512, int(currentT_R[2,3]) + 512), 2, (0,0,255), 2)
        odom = cv2.circle(odom, (int(x) + 512, int(z) + 512), 2, (0,255,0), 2)
        
        
        previous = current
        
        
        if frameNo:
            for i in range(len(current.points3D)):
                current.img_L = cv2.circle(current.img_L, tuple(matchesL[i]), 2, (255,0,0), 1)
                current.img_L = cv2.putText(current.img_L, str(round(current.points3D[i][2], 2)), (matchesL[i][0] + 5, matchesL[i][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
            
            cv2.imshow("Left Image", current.img_L)
            videoWriter.write(current.img_L)
        
        frameNo = frameNo + 1
        cv2.imshow("Map", odom)
        key_pressed = cv2.waitKey(1) & 0xFF
        
        if key_pressed == ord('q') or key_pressed == 27:
            break



def DesMatch(img_L, kp_l, des_l, img_R, kp_r, des_r, sort = True):
    # create BFMatcher object
    bfObj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # Match descriptors.
    matches = bfObj.match(des_l,des_r)
    
    kpL = []
    kpR = []
    
    minDist = 10000
    maxDist = 0
    
    for i in range(len(matches)):
        dist = matches[i].distance
        if dist < minDist:
            minDist = dist
        if dist > maxDist:
            maxDist = dist
    
    good = []
    
    for i in range(len(matches)):
        if matches[i].distance <= max(2 * minDist, 30):
            kpL.append((int(kp_l[matches[i].queryIdx].pt[0]), int(kp_l[matches[i].queryIdx].pt[1])))
            kpR.append((int(kp_r[matches[i].trainIdx].pt[0]), int(kp_r[matches[i].trainIdx].pt[1])))
            good.append(matches[i])
            
    kpR = np.array(kpR)
    kpL = np.array(kpL)
    
    return kpR, kpL, good


def findWorldPts(kpR, kpL, focalLength, baseLength, cx, cy):

    
    points3D = []
    matchesL = []
    matchesR = []
    for i in range(len(kpR)):
        
        d = kpL[i][0] - kpR[i][0]
        
        parallelCheck = kpL[i][1] - kpR[i][1]
        
        
        if (d > 0 and abs(parallelCheck) < 10):
            calcZ = (focalLength * baseLength) / d # Z = f*b/x1-x2
            Z = calcZ
            matchesL.append(kpL[i])
            matchesR.append(kpR[i])
            #print(kpR[i][0], kpL[i][0])
            
            X = np.abs(kpL[i][0]-cx) * calcZ / focalLength # X = x1*Z/f
            Y = np.abs(kpL[i][1]-cy) * calcZ / focalLength # Y = y1*Z/f
            
            if (Z > 5):
                points3D.append([X, Y, Z])
        
    return np.array(points3D), matchesL, matchesR

def poseEstimation(previous, current, focalLength, baseLength , cx , cy, prevT_L, prevT_R, K, scale):
    
    kpR, kpL, good = DesMatch(previous.img_L, previous.kp_l, previous.des_l, previous.img_R, previous.kp_r, previous.des_r)
            
    previous.points3D, matchesL, matchesR = findWorldPts(kpR, kpL, focalLength, baseLength , cx , cy)
    
    kpR, kpL, good = DesMatch(previous.img_L, previous.kp_l, previous.des_l, current.img_L, current.kp_l, current.des_l)
    
    
    E, mask = cv2.findEssentialMat(kpR, kpL, focalLength, (cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    _, R, t, mask = cv2.recoverPose(E, kpR, kpL, focal=focalLength, pp = (cx,cy))
    
    kpR, kpL, good = DesMatch(previous.img_R, previous.kp_r, previous.des_r, current.img_R, current.kp_r, current.des_r)
    
    
    R_prev = prevT_L[0:3, 0:3]
    t_prev = np.reshape(prevT_L[0:3, 3], (3,1))
    
    R_curr = R @ R_prev
    t_curr = t_prev + scale*(R_prev @ t)
    
    
    currentT_L = np.block([[R_curr, t_curr],[0, 0, 0, 1]])
    
    E, mask = cv2.findEssentialMat(kpR, kpL, focalLength, (cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    _, R, t, mask = cv2.recoverPose(E, kpR, kpL, focal=focalLength, pp = (cx,cy))
    
    
    kpR, kpL, good = DesMatch(current.img_L, current.kp_l, current.des_l, current.img_R, current.kp_r, current.des_r)
    
    
    current.points3D, matchesL, matchesR = findWorldPts(kpR, kpL, focalLength, baseLength , cx , cy)
    
    R_prev = prevT_R[0:3, 0:3]
    t_prev = np.reshape(prevT_R[0:3, 3], (3,1))
    
    R_curr = R @ R_prev
    t_curr = t_prev + scale*(R_prev @ t)
    
    
    currentT_R = np.block([[R_curr, t_curr],[0, 0, 0, 1]])
    
    return currentT_L, currentT_R, kpL, matchesL, matchesR

    
    
def getAbsoluteScale(frameNo, t):
    ss = t[frameNo-1].strip().split()
    x_prev = float(ss[3])
    y_prev = float(ss[7])
    z_prev = float(ss[11])
    ss = t[frameNo].strip().split()
    x = float(ss[3])
    y = float(ss[7])
    z = float(ss[11])
    return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev)), x, y, z
    
if __name__ == "__main__":
    main()
