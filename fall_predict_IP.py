__author__ = 'User'

import cv2
import urllib2
import numpy as np
import threading
import thread
from sklearn.lda import LDA
import math


# url_cam1 = "http://192.168.1.107/video1.h264"
# url_cam2 = "http://192.168.1.108/video1.h264"
# url_cam3 = "http://192.168.1.104/video1.h264"
# url_cam4 = "http://192.168.1.106/video1.h264"

url_cam1 = "http://192.168.0.100/video2.jpeg"
url_cam2 = "http://192.168.0.103/video2.jpeg"
url_cam3 = "http://192.168.0.102/video1.jpeg"
url_cam4 = "http://192.168.0.101/video2.jpeg"

name = "farm"
time = 6500
# act = "BG"            #time 1200
# act = "-walk"         #time 2600
act = "-act"          #time 6500
# act = "-sit"          #time 11500
# act = "-couch"
# act = "-layS"
# act = "-layF"
# act = "-Ffore"
# act = "-Fback"
# act = "-Fside"
# act = "-Fsit"

fourcc = cv2.cv.CV_FOURCC('I','Y','U','V')
outN1 = "E:/Video_final/predict/"+name+"_cam1"+act+".avi"
out1 = cv2.VideoWriter(outN1,fourcc, 20, (640, 360))
outN2 = "E:/Video_final/predict/"+name+"_cam2"+act+".avi"
out2 = cv2.VideoWriter(outN2,fourcc, 20, (640, 360))
outN3 = "E:/Video_final/predict/"+name+"_cam3"+act+".avi"
out3 = cv2.VideoWriter(outN3,fourcc, 20, (640, 360))
outN4 = "E:/Video_final/predict/"+name+"_cam4"+act+".avi"
out4 = cv2.VideoWriter(outN4,fourcc, 20, (640, 360))

#prepare for predict
X = np.zeros((25,3), np.float)
X[0:5,:] = np.loadtxt('fall_detect_data_for_analyze/walk_data.txt')
X[5:10,:] = np.loadtxt('fall_detect_data_for_analyze/sit_data.txt')
X[10:15,:] = np.loadtxt('fall_detect_data_for_analyze/lie_down_floor_data.txt')
X[15:20,:] = np.loadtxt('fall_detect_data_for_analyze/lie_still_floor_data.txt')
X[20:25,:] = np.loadtxt('fall_detect_data_for_analyze/fall_data.txt')

Y = np.zeros(25, np.int)
Y[0:5] = 1
Y[5:10] = 2
Y[10:15] = 3
Y[15:20] = 4
Y[20:25] = 5

sklearn_lda = LDA(n_components=None)
X_lda_sklearn = sklearn_lda.fit_transform(X, Y)

class open_cam(threading.Thread):
    def run(self):
        print "Thread Number: ", thread.get_ident()
        print  self.url

        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()

        password_mgr.add_password(None, self.url, "admin", "1234567")

        handler_cam = urllib2.HTTPBasicAuthHandler(password_mgr)

        opener_cam = urllib2.build_opener(handler_cam)
        opener_cam.open(self.url)
        urllib2.install_opener(opener_cam)

        stream_cam=urllib2.urlopen(self.url)

        t = 0
        aspect = np.zeros(16, np.float)
        j = 0

        #backgroundsubtraction
        BG = cv2.BackgroundSubtractorMOG2(500,32,True)

        bytes=''
        while True:
            bytes+=stream_cam.read(1024)
            a = bytes.find('\xff\xd8')
            b = bytes.find('\xff\xd9')
            if a!=-1 and b!=-1:
                jpg = bytes[a:b+2]
                bytes= bytes[b+2:]
                cam = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
                window_cam = self.url
                # self.out.write(cam)

                if t>=1:
                    print t
                    #foreground mask
                    # maskN = np.zeros(cam.shape, np.uint8)
                    median = cv2.medianBlur(cam,3)
                    color_image = cv2.GaussianBlur(median, (7,7),5)
                    mask = BG.apply(color_image)
                    maskN = getmask(mask)
                    contours, hierarchy = cv2.findContours(maskN,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    areas = [cv2.contourArea(c) for c in contours]

                    tempAR = aspect
                    aspect = np.zeros(16, np.float)
                    aspect[0:15] = tempAR[1:16]
                    if not areas == [] :
                        # Find the index that contour in range
                        temp = 0
                        te = 0
                        for k in range(len(areas)):
                            if areas[k] > 1400:
                                if areas[k]>temp:
                                    temp = areas[k]
                                    te = k
                        # i+=1

                        if temp != 0:
                            cnt = contours[te]
                            #rect = cv2.minAreaRect(cnt)
                            #box = cv2.cv.BoxPoints(rect)
                            #box = np.int0(box)
                            #cv2.drawContours(cam,[box],0,(0,0,125),2)
                            # ((x,y),(w,h),angle[i]) = rect
                            x,y,w,h = cv2.boundingRect(cnt)
                            cv2.rectangle(cam,(x,y),(x+w,y+h),(0,255,0),2)
                            aspect[15] = float(w)/h
                            #cv2.circle(cam,(x+w/2,y+h/2),2,(255,0,0),2)

                            angle = 0.0
                            if len(cnt) >= 5:
                                ((x1,y1),(MA,ma),angle) = cv2.fitEllipse(cnt)
                                # cv2.ellipse(cam,((x1,y1),(MA,ma),angle),(0,185,0),2)

                            # N = y - (math.tan(angle)*x)

                            # theta = abs(angle-90)*3.14/180.0
                            # x2 = int(round(x+w/2 + 80 * math.cos(theta)))
                            # y2 = int(round(y+h/2 + 80 * math.sin(theta)))
                            # cv2.line(cam,(x+w/2,y+h/2),(x2,y2),(140,0,0),2)
                            if t > 15:
                                action = "???"
                                changeA = ratechange_aspect(aspect)
                                predict = sklearn_lda.predict([aspect[15],changeA[15],abs(angle-90)])
                                if predict == 1:
                                    action = "Walk"
                                elif predict == 2:
                                    action = "Sit"
                                elif predict == 3:
                                    action = "Lie Down"
                                elif predict == 4:
                                    action = "Lie Still"
                                elif predict == 5:
                                    action ="Fall"
                                    cv2.circle(cam,(550,40),20,(0,0,255),-1)
                                    # cv2.putText(cam, 'FALL',(500,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),1)
                                # else:
                                #     j=0
                                cv2.putText(cam,action,(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(0,0,200),1)
                            else:
                                cv2.putText(cam, 'Not Found',(500,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),1)


                cv2.imshow(window_cam,cam)
                # self.out.write(cam)
                t = t + 1

                key = cv2.waitKey(1)
                if key in [27, ord('Q'), ord('q')]: # exit on ESC
                    break
                # elif t > time:
                #     print "Finished :]" + self.url
                #     break




def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def ratechange_aspect(aspect, window=15):
    change_aspect = np.zeros(aspect.shape, np.float)
    for j in range(window, len(aspect)):
        if aspect[j] != 0:
            change_aspect[j] = (aspect[j]-aspect[j-window])/window
    return change_aspect

def getmask(img):

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    maskN = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel, iterations = 1)
    maskN = cv2.dilate(maskN,kernel,iterations=2)
    maskN = cv2.morphologyEx(maskN,cv2.MORPH_CLOSE,kernel, iterations = 5)
    _, maskN = cv2.threshold(maskN, 127, 255, cv2.THRESH_BINARY)
    maskN[0:40,0:250] = [0]

    return maskN

def main():
    thr1 = open_cam()
    thr1.url = url_cam1
    thr1.out = out1

    thr2 = open_cam()
    thr2.url = url_cam2
    thr2.out = out2

    thr3 = open_cam()
    thr3.url=url_cam3
    thr3.out = out3

    thr4 = open_cam()
    thr4.url=url_cam4
    thr4.out = out4

    thr1.start()
    thr2.start()
    thr3.start()
    thr4.start()

main()
