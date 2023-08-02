import cv2 as cv
import numpy as np
import os
import time
import HandTrackingModule as htm

def resizeFrame(frame,scale=0.75):
    height=int(frame.shape[0]*scale)
    width=int(frame.shape[1]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

path=os.path.dirname(__file__)+"/fingerImages"
mylist=os.listdir(path)

overlayList=[]
for imPath in mylist:
    image=resizeFrame(cv.imread(f'{path}/{imPath}'),0.40)
    overlayList.append(image)
# print(overlayList)

wCam,hCam=640,480

ctime=0
ptime=0

detector=htm.HandDetector(detectionCon=0.75,maxHands=1)

cap=cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

while(True):
    success,img=cap.read()

    img=detector.findHands(img)
    lmList=detector.findPosition(img,False)
    # print(lmList)
    imgNo=0
    if len(lmList)!=0:
        n=0
        u=1
        v=1
        z=0
        # if (lmList[0][2]<lmList[5][2] or lmList[0][2]>lmList[17][2]):
            # z=1
        # if z==0:
        if lmList[5][1]<lmList[17][1]:
            u=-1
        if lmList[0][2]<lmList[9][2]:
            v=-1
        for i in range(4,21,4):
            if(i==4):
                if(u*lmList[i][1]>u*lmList[i-2][1]):
                    n+=1
                continue
            if(v*lmList[i][2]<v*lmList[i-2][2]):
                n+=1
        
        # else:            
        #     if lmList[5][2]<lmList[17][2]:
        #         u=-1
        #     if lmList[0][1]<lmList[9][1]:
        #         v=-1
        #     for i in range(4,21,4):
        #         if(i==4):
        #             if(u*lmList[i][2]>u*lmList[i-2][2]):
        #                 n+=1
        #             continue
        #         if(v*lmList[i][1]<v*lmList[i-2][1]):
        #             n+=1
        # print(n)
        h,w,c=overlayList[n].shape
        img[0:h,0:w]=overlayList[n]

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,f'FPS:{int(fps)}',(500,50),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

    cv.imshow("Video",img)
    if cv.waitKey(1) & 0xff==ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()


