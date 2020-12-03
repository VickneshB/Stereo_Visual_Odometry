import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
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
        
    def DesMatch(self, img_L, kp_l, des_l, img_R, kp_r, des_r, sort = True):
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


    def findWorldPts(self, kpR, kpL, focalLength, baseLength, cx, cy):

        
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

    def poseEstimation(self, previous, current, focalLength, baseLength , cx , cy, prevT_L, prevT_R, K, scale):
        
        kpR, kpL, good = self.DesMatch(previous.img_L, previous.kp_l, previous.des_l, previous.img_R, previous.kp_r, previous.des_r)
                
        previous.points3D, matchesL, matchesR = self.findWorldPts(kpR, kpL, focalLength, baseLength , cx , cy)
        
        kpR, kpL, good = self.DesMatch(previous.img_L, previous.kp_l, previous.des_l, current.img_L, current.kp_l, current.des_l)
        
        
        E, mask = cv2.findEssentialMat(kpR, kpL, focalLength, (cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        _, R, t, mask = cv2.recoverPose(E, kpR, kpL, focal=focalLength, pp = (cx,cy))
        
        kpR, kpL, good = self.DesMatch(previous.img_R, previous.kp_r, previous.des_r, current.img_R, current.kp_r, current.des_r)
        
        
        R_prev = prevT_L[0:3, 0:3]
        t_prev = np.reshape(prevT_L[0:3, 3], (3,1))
        
        R_curr = R @ R_prev
        t_curr = t_prev + scale*(R_prev @ t)
        
        
        currentT_L = np.block([[R_curr, t_curr],[0, 0, 0, 1]])
        
        E, mask = cv2.findEssentialMat(kpR, kpL, focalLength, (cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        _, R, t, mask = cv2.recoverPose(E, kpR, kpL, focal=focalLength, pp = (cx,cy))
        
        
        kpR, kpL, good = self.DesMatch(current.img_L, current.kp_l, current.des_l, current.img_R, current.kp_r, current.des_r)
        
        
        current.points3D, matchesL, matchesR = self.findWorldPts(kpR, kpL, focalLength, baseLength , cx , cy)
        
        R_prev = prevT_R[0:3, 0:3]
        t_prev = np.reshape(prevT_R[0:3, 3], (3,1))
        
        R_curr = R @ R_prev
        t_curr = t_prev + scale*(R_prev @ t)
        
        
        currentT_R = np.block([[R_curr, t_curr],[0, 0, 0, 1]])
        
        return currentT_L, currentT_R, kpL, matchesL, matchesR
