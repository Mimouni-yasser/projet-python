import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from time import sleep
import concurrent.futures
import subprocess
import sys
 


url='http://192.168.71.51:1024/shot.jpg'
im=None
stop_data = cv2.CascadeClassifier('./haar/stop.xml') #ces fichiers contient le data necessaire pour la reconnaissances des panneaux et du feu rouge
light_data = cv2.CascadeClassifier('./haar/light.xml')
global x_medium
global y_medium 


f = open("./server/index.html", 'w')

def detect_signs():
    global detected_stop
    global detected_light
    detected_stop = False
    detected_light = False
    count_stop = 0
    count_light = 0
    while True:
        imgResp = urllib.request.urlopen(url) #capter une image
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
        img = cv2.imdecode(imgNp,-1) #l'image capter sera un tableau binaire, c'est 2 lignes converti ce dernier en image presentable

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #l'algorithme utilise l'image en grayscale et en RGB, ainsi ces 2 lignes
        found_stop = stop_data.detectMultiScale(img_gray, 
                                           minSize =(50, 50))   #detection..
        found_light = light_data.detectMultiScale(img_gray, 
                                           minSize =(50, 50))
        amount_found = len(found_stop) + len(found_light)
              
        if amount_found != 0:
              #tracer des rectangles autour les objs detecter, ghir pour demo, a retrancher dans le code final
            if(len(found_light)>0):
                count_light +=1
                if count_light > 10:    
                    detected_light = True
                    print("red")
                    f.seek(0)
                    f.write("red light")
                    f.truncate()
                    sleep(3)
                    detected_light = False

            for (x, y, width, height) in found_light:
                cv2.rectangle(img, (x, y), 
                              (x + height, y + width), 
                              (255, 0, 0), 5)
            if(len(found_stop)>0):
                count +=1
                if count > 10:
                    detected_stop = True
                    print("stop")
                    f.seek(0)
                    f.write("stop")
                    f.truncate()
                    sleep(3)
                    detected_stop = False
            else:
                count = 0
            for (x, y, width, height) in found_stop:
                cv2.rectangle(img, (x, y), 
                              (x + height, y + width), 
                              (0, 0, 255), 5)
        cv2.imshow("sign + light", img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyWindow('sign + light')
    return

def detect_line():
    (x, y, w, h) = (0,0,0,0)
    count_left = 0
    count_right = 0
    count_forward = 0
    while True:
        imgResp = urllib.request.urlopen(url) #capter un image ...
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
        frame = cv2.imdecode(imgNp,-1)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # red color
        low_black  = np.array([0, 0, 0])
        high_black = np.array([ 50, 50,50]) #definir les limites des colors a detecter, ici de noir au gris foncé
        black_mask = cv2.inRange(hsv_frame, low_black, high_black) #couper les autres colours

        contours, _ = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #trouvez la colour 
        contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True) #triez les lieu par ordre de surface
        if len(contours) > 1:
            (x, y, w, h) = cv2.boundingRect(contours[0]) #tracer rectangle

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x_medium = int((x + x + w) / 2)
        y_medium = int((y + y + h) / 2)
        cv2.imshow("Frame", frame)
        #cv2.imshow("mask", black_mask) #enfin afficher l'image et le mask
        (_,_,w_frame,h_frame)=cv2.getWindowImageRect('Frame')
        if not detected_stop and not detected_light:
            if x_medium>0.75*w_frame:
                    print('right')
                    f.seek(0)
                    f.write('right')
                    f.truncate()
                    sleep(0.001)

            elif x_medium<0.5*w_frame:
                    print('left')
                    f.seek(0)
                    f.write('left')
                    f.truncate()
                    sleep(0.01)
            else:
                    print('forward')
                    f.seek(0)
                    f.write('forward')
                    f.truncate()
                    sleep(0.01)


        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyWindow('frame')
    cv2.destroyWindow('mask')
    return

 
 
if __name__ == '__main__':
    print("started")
    #server = subprocess.Popen(['python', '-m','http.server', '--directory', './server'], creationflags=subprocess.CREATE_NEW_CONSOLE)
    with concurrent.futures.ThreadPoolExecutor(max_workers = 4) as executer:
            f1= executer.submit(detect_line)
            f2= executer.submit(detect_signs)
            print(f1.result())
            print("--------------------------------------------------------------------------------------------------------------------------------------")
            print(f2.result())

    f.close();