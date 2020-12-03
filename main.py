from Stereo import *
        
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
    
    sequence = "01"   
    calibFile = open(datasetDir + sequence + "/calib.txt","r")
    timeFile = open("dataset/poses/" + sequence + ".txt","r")
    projectionMatrices = calibFile.read().split()
    
    focalLength = float(projectionMatrices[1])
    baseLength =  -1*float(projectionMatrices[17])/float(projectionMatrices[14]) # base = -P1(1,4)/P1(1,1) (in meters)
    
    P1 = [[float(projectionMatrices[1]), float(projectionMatrices[2]), float(projectionMatrices[3]), 0.00],
                       [float(projectionMatrices[5]), float(projectionMatrices[6]), float(projectionMatrices[7]), 0.00],
                       [float(projectionMatrices[9]), float(projectionMatrices[10]), float(projectionMatrices[11]), 0.00]]
    
    P1 = np.array(P1)
    
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
            currentT, matchesL, matchesR = current.poseEstimation(previous, current, focalLength, baseLength , cx , cy, currentT, P1)
            
        print(currentT[:3,3])
        
        odom = cv2.circle(odom, (int(currentT[0,3]) + 512, int(currentT[2,3]) + 512), 2, (255,0,0), 2)
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

    
if __name__ == "__main__":
    main()
