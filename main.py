from Stereo import *
        
def getGroundTruth(frameNo, t):
    ss = t[frameNo].strip().split()
    x = float(ss[3])
    y = float(ss[7])
    z = float(ss[11])
    return x, y, z
    

def main():
    
    fname = "DepthImage.avi"
    try:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fps = 15.0
        videoWriter = cv2.VideoWriter(fname, fourcc, fps, (1241, 376))
    except:
        print("Error: can't create output video: %s" % fname)
        sys.exit()
    
    maxMatches = 500
    frameNo = 0
    
    currentT = np.eye(4)
    
    odom = np.zeros((1024,1024,3))
    kpR = []
    datasetDir = "dataset/sequences/"
    
    sequence = "00"   
    calibFile = open(datasetDir + sequence + "/calib.txt","r")
    timeFile = open("dataset/poses/" + sequence + ".txt","r")
    projectionMatrices = calibFile.read().split()
    
    focalLength = float(projectionMatrices[1])
    baseLength =  -1*float(projectionMatrices[17])/float(projectionMatrices[14]) # base = -P1(1,4)/P1(1,1) (in meters)
    
    P1 = [[float(projectionMatrices[1]), float(projectionMatrices[2]), float(projectionMatrices[3]), float(projectionMatrices[4])],
                      [float(projectionMatrices[5]), float(projectionMatrices[6]), float(projectionMatrices[7]), float(projectionMatrices[8])],
                      [float(projectionMatrices[9]), float(projectionMatrices[10]), float(projectionMatrices[11]), float(projectionMatrices[12])]]
    
    P1 = np.array(P1)
    
    P2 = [[float(projectionMatrices[14]), float(projectionMatrices[15]), float(projectionMatrices[16]), float(projectionMatrices[17])],
                      [float(projectionMatrices[18]), float(projectionMatrices[19]), float(projectionMatrices[20]), float(projectionMatrices[21])],
                      [float(projectionMatrices[22]), float(projectionMatrices[23]), float(projectionMatrices[24]), float(projectionMatrices[25])]]
    
    P2 = np.array(P2)
    
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
        
        x, y, z = getGroundTruth(frameNo, t)
        
        if frameNo:
            currentT = current.poseEstimation(previous, current, focalLength, baseLength , cx , cy, currentT, P1, P2)
            
        print(currentT[:3,3])
        
        odom = cv2.circle(odom, (int(currentT[0,3]) + 256, int(currentT[2,3]) + 256), 2, (255,0,0), 2)
        odom = cv2.circle(odom, (int(x) + 256, int(z) + 256), 2, (0,255,0), 2)
        
        
        previous = current
            
        cv2.imshow("Left Image", current.img_L)
        videoWriter.write(current.img_L)
        
        frameNo = frameNo + 1
        cv2.imshow("Map", odom)
        key_pressed = cv2.waitKey(1) & 0xFF
        
        if key_pressed == ord('q') or key_pressed == 27:
            break

    
if __name__ == "__main__":
    main()
