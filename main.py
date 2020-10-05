import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

def main():
    
    
    maxMatches = 10
    
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
            IMG_L = cv2.imread(sequence + "/image_0/" + image)
            IMG_R = cv2.imread(sequence + "/image_1/" + image)
            
            h, w, _ = IMG_L.shape
            #print(h, w)
            
            kp_l, des_l = KpDes(IMG_L)
            kp_r, des_r = KpDes(IMG_R)
            
            IMG_L1 = cv2.drawKeypoints(IMG_L, kp_l, None, color = (0,255,0), flags = 0)
            IMG_R1 = cv2.drawKeypoints(IMG_R, kp_r, None, color = (0,255,0), flags = 0)
            bestMatches_Train, bestMatches_Query, matchIMG = DesMatch(IMG_L, kp_l, des_l, IMG_R, kp_r, des_r, maxMatches)
            
            X, Y, Z = findWorldPts(bestMatches_Train, bestMatches_Query, focalLength, baseLength , cx , cy)
            
            '''for i in range(len(bestMatches_Train)):
                IMG_L = cv2.circle(IMG_L, (bestMatches_Train[i]), 2, (255,0,0), 1)
                IMG_L = cv2.putText(IMG_L, str(Z[i]), (bestMatches_Train[i][0] + 5, bestMatches_Train[i][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .3, (0,0,255), 1, cv2.LINE_AA)'''
            
            cv2.imshow("Left Image", IMG_L)
            # cv2.imshow("Right Image", IMG_R)
            # cv2.imshow("Match Image", matchIMG)
            # cv2.waitKey(0)
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

def DesMatch(IMG_L, kp_l, des_l, IMG_R, kp_r, des_r, maxMatches):
    # create BFMatcher object
    bfObj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bfObj.match(des_l,des_r)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    bestMatches_Train = []
    bestMatches_Query = []
    for match in matches:
        #print(match)
        #print(kp_l[match.trainIdx], kp_r[match.queryIdx])
        bestMatches_Query.append((int(kp_l[match.queryIdx].pt[0]), int(kp_l[match.queryIdx].pt[1])))
        bestMatches_Train.append((int(kp_r[match.trainIdx].pt[0]), int(kp_r[match.trainIdx].pt[1])))
        if len(bestMatches_Query) == maxMatches:
            break
        
    #print(bestMatches_Train[0][0], bestMatches_Query[0][0])
        
    # Draw first 100 matches.
    IMG = cv2.drawMatches(IMG_L, kp_l, IMG_R, kp_r, matches[:maxMatches], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return bestMatches_Train, bestMatches_Query, IMG


def findWorldPts(bestMatches_Train, bestMatches_Query, focalLength, baseLength, cx, cy):

    X = []
    Y = []
    Z = []
    for i in range(len(bestMatches_Train)):
        
        calcZ = (focalLength * baseLength) / ((bestMatches_Query[i][0] - cx) - (bestMatches_Train[i][0] - cx)) # Z = f*b/x1-x2
        Z.append(calcZ)
        
        #print(bestMatches_Train[i][0], bestMatches_Query[i][0])
        
        X.append(np.abs(bestMatches_Train[i][0]-cx) * calcZ / focalLength) # X = x1*Z/f
        Y.append(np.abs(bestMatches_Train[i][1]-cy) * calcZ / focalLength) # Y = y1*Z/f
    
    return X, Y, Z
    
if __name__ == "__main__":
    main()
