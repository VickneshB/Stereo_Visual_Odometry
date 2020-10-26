import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from icp import icp

class Stereo:
    
    IMG_L = []
    IMG_R = []
    kp_l = []
    kp_r = []
    des_l = []
    des_r = []
    
    def __init__(self, frameNo):
        self.frameNo = frameNo
    
    def Matching(self, IMG_L, IMG_R):
        self.IMG_L = IMG_L
        self.IMG_R = IMG_R
        self.kp_l, self.des_l = KpDes(IMG_L)
        self.kp_r, self.des_r = KpDes(IMG_R)
    

def main():
    
    
    maxMatches = 50
    frameNo = 0
    
    currentTransformation = np.eye(4)
    
    mapImage = np.zeros((1024,1024))
    
    datasetDir = "dataset/sequences/*"
    for sequence in glob.glob(datasetDir):
        
        calibFile = open(sequence + "/calib.txt","r")
        #print(calibFile.read().split()[17])#, calibFile.read().split()[15], calibFile.read().split()[18])
        projectionMatrices = calibFile.read().split()
        
        focalLength = float(projectionMatrices[1])
        baseLength =  -1*float(projectionMatrices[17])/float(projectionMatrices[14]) # base = -P1(1,4)/P1(1,1) (in meters)
        
        intrinsicMatrix = [[float(projectionMatrices[1]), float(projectionMatrices[2]), float(projectionMatrices[3])],
                           [float(projectionMatrices[5]), float(projectionMatrices[6]), float(projectionMatrices[7])],
                           [float(projectionMatrices[9]), float(projectionMatrices[10]), float(projectionMatrices[11])]]
        
        cx = float(projectionMatrices[3])
        cy = float(projectionMatrices[7])
        
        #print(intrinsicMatrix)
        #print(focalLength, baseLength)
        
        imageNames = os.listdir(sequence + "/image_0")
        imageNames.sort()
        for image in imageNames:
            current = Stereo(frameNo)
            
            imgL = cv2.imread(sequence + "/image_0/" + image)
            imgR = cv2.imread(sequence + "/image_1/" + image)
            
            #h, w, _ = imgL.shape
            #print(h, w)
            
            current.Matching(imgL, imgR)
            
            #kp_l, des_l = KpDes(IMG_L)
            #kp_r, des_r = KpDes(IMG_R)
            
            #print(current.kp_l)
            
            #IMG_L1 = cv2.drawKeypoints(IMG_L, kp_l, None, color = (0,255,0), flags = 0)
            #IMG_R1 = cv2.drawKeypoints(IMG_R, kp_r, None, color = (0,255,0), flags = 0)
            
            
            if frameNo:
                currentTransformation = poseEstimation(previous, current, maxMatches, focalLength, baseLength , cx , cy, currentTransformation)
                
            print(currentTransformation[:3,3])
            
            #mapImage[int(currentTransformation[2,3]) + 512, int(currentTransformation[0,3]) + 512] = 255
            
            mapImage = cv2.circle(mapImage, (int(currentTransformation[0,3]) + 512, int(currentTransformation[2,3]) + 512), 2, (255,0,0), 2)
               
            previous = current
            
            #previous3Dpoints = current3Dpoints
            
            #print(current3Dpoints)
            
            '''for i in range(len(bestMatches_Train)):
                IMG_L = cv2.circle(IMG_L, (bestMatches_Train[i]), 2, (255,0,0), 1)
                IMG_L = cv2.putText(IMG_L, str(current3Dpoints[i][2]), (bestMatches_Train[i][0] + 5, bestMatches_Train[i][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .3, (0,0,255), 1, cv2.LINE_AA)'''
            
            frameNo = frameNo + 1
            cv2.imshow("Left Image", current.IMG_L)
            # cv2.imshow("Right Image", IMG_R)
            # cv2.imshow("Match Image", matchIMG)
            # cv2.waitKey(0)
            cv2.imshow("Map", mapImage)
            key_pressed = cv2.waitKey(1) & 0xFF
            
            if key_pressed == ord('q') or key_pressed == 27:
                break
        if key_pressed == ord('q') or key_pressed == 27:
                break

def KpDes(IMG):
        
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and compute the descriptors with ORB
    kp, des = orb.detectAndCompute(IMG,None)
    
    return kp, des

def DesMatch(IMG_L, kp_l, des_l, IMG_R, kp_r, des_r, maxMatches = 100, sort = True):
    # create BFMatcher object
    bfObj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    #print(type(des_l))
    matches = bfObj.match(des_l,des_r)
    # Sort them in the order of their distance.
    if sort:
        matches = sorted(matches, key = lambda x:x.distance)
    
    bestMatches_Train = []
    bestMatches_Query = []
    descriptor_Train = []
    descriptor_Query = []
    keypoints_Train = []
    keypoints_Query = []
    
    for match in matches:
        #print(match)
        #print(kp_l[match.trainIdx], kp_r[match.queryIdx])
        bestMatches_Query.append((int(kp_l[match.queryIdx].pt[0]), int(kp_l[match.queryIdx].pt[1])))
        bestMatches_Train.append((int(kp_r[match.trainIdx].pt[0]), int(kp_r[match.trainIdx].pt[1])))
        keypoints_Query.append(kp_l[match.queryIdx])
        keypoints_Train.append(kp_r[match.trainIdx])
        descriptor_Query.append(des_l[match.queryIdx])
        descriptor_Train.append(des_r[match.trainIdx])
        if len(bestMatches_Query) == maxMatches:
            break
        
    #print(bestMatches_Train[0][0], bestMatches_Query[0][0])
    #keypoints_Train = np.array(bestMatches_Train)
    #keypoints_Query = np.array(bestMatches_Query)
    descriptor_Train = np.array(descriptor_Train)
    descriptor_Query = np.array(descriptor_Query)
    
    #print(type(keypoints_Train), type(keypoints_Query[0]))
    #print(type(descriptor_Train), type(descriptor_Query[0]))
    
    # Draw first 100 matches.
    #IMG = cv2.drawMatches(IMG_L, kp_l, IMG_R, kp_r, matches[:maxMatches], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv2.imshow("Match Image", IMG)
    #cv2.waitKey(0)
    
    return bestMatches_Train, bestMatches_Query, descriptor_Train, descriptor_Query, keypoints_Train, keypoints_Query


def findWorldPts(bestMatches_Train, bestMatches_Query, focalLength, baseLength, cx, cy):

    
    current3Dpoints = []
    for i in range(len(bestMatches_Train)):
        
        d = bestMatches_Query[i][0] - bestMatches_Train[i][0]
        
        parallelCheck = bestMatches_Query[i][1] - bestMatches_Train[i][1]
        
        if (d > 0 && abs(parallelCheck) < 10):
            calcZ = (focalLength * baseLength) / d # Z = f*b/x1-x2
            Z = calcZ
            
            #print(bestMatches_Train[i][0], bestMatches_Query[i][0])
            
            X = np.abs(bestMatches_Query[i][0]-cx) * calcZ / focalLength # X = x1*Z/f
            Y = np.abs(bestMatches_Query[i][1]-cy) * calcZ / focalLength # Y = y1*Z/f
            
            current3Dpoints.append([X, Y, Z])
        
    return np.array(current3Dpoints)

def poseEstimation(previous, current, maxMatches, focalLength, baseLength , cx , cy, currentTransformation):
    
    bestMatches_Train, bestMatches_Query, descriptor_Train, descriptor_Query, keypoints_Train, keypoints_Query = DesMatch(previous.IMG_L, previous.kp_l, previous.des_l, previous.IMG_R, previous.kp_r, previous.des_r, maxMatches)
            
    previous3Dpoints = findWorldPts(bestMatches_Train, bestMatches_Query, focalLength, baseLength , cx , cy)
    
    #bestMatches_Train, bestMatches_Query, descriptor_Train, descriptor_Query, keypoints_Train, keypoints_Query = DesMatch(previous.IMG_L, keypoints_Query, descriptor_Query, current.IMG_L, current.kp_l, current.des_l, maxMatches)
    
    bestMatches_Train, bestMatches_Query, descriptor_Train, descriptor_Query, keypoints_Train, keypoints_Query = DesMatch(current.IMG_L, keypoints_Train, descriptor_Train, current.IMG_R, current.kp_r, current.des_r)
    
    #print("QUERY\n", bestMatches_Query)
    #print("TRAIN\n", bestMatches_Train)
    
    current3Dpoints = findWorldPts(bestMatches_Train, bestMatches_Query, focalLength, baseLength , cx , cy)
    
    #print(previous3Dpoints.shape)
    #print(current3Dpoints.shape)
    
    #H = fitPoint(previous3Dpoints, current3Dpoints)
    T, distances, i = icp(previous3Dpoints, current3Dpoints)
    
    currentTransformation = currentTransformation @ T
    
    return currentTransformation

def fitPoint(previous3Dpoints, current3Dpoints):
    
    centroidPrevious = np.mean(previous3Dpoints, axis = 0)
    centroidCurrent = np.mean(current3Dpoints, axis = 0)
    
    Previous = previous3Dpoints - centroidPrevious
    Current = current3Dpoints - centroidCurrent
    
    W = np.dot(Previous.T, Current)
    
    U, S, Vt = np.linalg.svd(W)
    
    R = np.dot(Vt.T, U.T)
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
        
    t = centroidCurrent.T - np.dot(R, centroidPrevious.T)
    
    H = np.eye(4)
    H[:3,:3] = R
    H[:3,3] = t
    #print(H)
    
    return H
    
if __name__ == "__main__":
    main()
