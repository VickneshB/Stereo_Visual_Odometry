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
        
    def DesMatch(self, img_L, kp_l, des_l, img_R, kp_r, des_r, mono_points3D = None):
        # create BFMatcher object
        bfObj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # Match descriptors.
        matches = bfObj.match(des_l,des_r)
        
        pts_L = []
        pts_R = []
        desL = []
        desR = []
        kpL = []
        kpR = []
        
        minDist = 10000
        maxDist = 0
        
        # Finding the good matches
        for i in range(len(matches)):
            dist = matches[i].distance
            if dist < minDist:
                minDist = dist
            if dist > maxDist:
                maxDist = dist
        
        good = []
        mono = []
        
        for i in range(len(matches)):
            if matches[i].distance <= max(2 * minDist, 30):
                pts_L.append(kp_l[matches[i].queryIdx].pt)
                pts_R.append(kp_r[matches[i].trainIdx].pt)
                kpL.append(kp_l[matches[i].queryIdx])
                kpR.append(kp_r[matches[i].trainIdx])
                desL.append(des_l[matches[i].queryIdx])
                desR.append(des_r[matches[i].trainIdx])
                
                if mono_points3D is not None:
                    mono.append(mono_points3D[matches[i].queryIdx])
                good.append(matches[i])
                
        pts_L = np.array(pts_L)
        pts_R = np.array(pts_R)
        kpL = np.array(kpL)
        kpR = np.array(kpR)
        desL = np.array(desL)
        desR = np.array(desR)
        mono = np.array(mono)
        
        return pts_L, pts_R, kpL, kpR, desL, desR, good, mono


    def findWorldPts(self, pts_L, pts_R, focalLength, baseLength, cx, cy):

        
        points3D = []
        matchesL = []
        matchesR = []
        for i in range(len(pts_R)):
            
            d = pts_L[i][0] - pts_R[i][0]
            
            
            calcZ = (focalLength * baseLength) / d # Z = f*b/x1-x2
            Z = calcZ
            matchesL.append(pts_L[i])
            matchesR.append(pts_R[i])
            
            X = np.abs(pts_L[i][0]-cx) * calcZ / focalLength # X = x1*Z/f
            Y = np.abs(pts_L[i][1]-cy) * calcZ / focalLength # Y = y1*Z/f
            points3D.append([X, Y, Z])
            
        return np.array(points3D), matchesL, matchesR
    
    
    def computeScale(self, monoPoints, stereoPoints):
        
        scales = []

        for i in range(len(monoPoints)):
            
            monoMag = ((monoPoints[i][0])**2 + (monoPoints[i][1])**2 + (monoPoints[i][2])**2)**0.5
            stereoMag = ((stereoPoints[i][0])**2 + (stereoPoints[i][1])**2 + (stereoPoints[i][2])**2)**0.5
            
            scales.append(monoMag/stereoMag)
        
        return np.mean(scales)
        
    def poseEstimation(self, previous, current, focalLength, baseLength , cx , cy, prevT_L, P1, P2):
        
        pts_prev, pts_current, kpL, kpR, desL, desR, good, _ = self.DesMatch(previous.img_L, previous.kp_l, previous.des_l, current.img_L, current.kp_l, current.des_l)
        
        E, mask = cv2.findEssentialMat(pts_current, pts_prev, focalLength, (cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        _, R, t, _ = cv2.recoverPose(E, pts_current, pts_prev, focal=focalLength, pp = (cx,cy))
        

        pts_prev = pts_prev[mask.ravel() == 1]
        pts_current = pts_current[mask.ravel() == 1]
        kpL = kpL[mask.ravel() == 1]
        kpR = kpR[mask.ravel() == 1]
        desL = desL[mask.ravel() == 1]
        desR = desR[mask.ravel() == 1]
        
        P3 = P1.copy()
        P3[:3, :3] = P3[:3, :3] @ R
        P3[0:3,:] = t
        
        mono_points3D = cv2.triangulatePoints(P1, P3, pts_prev.T, pts_current.T)
        mono_points3D = mono_points3D.T
        
        
        pts_L, pts_R, kpL, kpR, desL, desR, good, mono_points3D = self.DesMatch(previous.img_L, kpL, desL, previous.img_R, previous.kp_r, previous.des_r, mono_points3D)
        
        E, mask = cv2.findEssentialMat(pts_R, pts_L, focalLength, (cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        pts_L = pts_L[mask.ravel() == 1]
        pts_R = pts_R[mask.ravel() == 1]
        kpL = kpL[mask.ravel() == 1]
        kpR = kpR[mask.ravel() == 1]
        desL = desL[mask.ravel() == 1]
        desR = desR[mask.ravel() == 1]
        mono_points3D = mono_points3D[mask.ravel() == 1]
        
        previous.points3D = cv2.triangulatePoints(P1, P2, pts_L.T, pts_R.T)
        previous.points3D = previous.points3D.T
        
        
        scale = self.computeScale(mono_points3D, previous.points3D)
        
        print("SCALE: ", scale)
        
        R_prev = prevT_L[0:3, 0:3]
        t_prev = np.reshape(prevT_L[0:3, 3], (3,1))
        
        R_curr = R_prev @ R
        t_curr = t_prev + scale*(R_prev @ t)
        
        
        currentT = np.block([[R_curr, t_curr],[0, 0, 0, 1]])
        
        return currentT
