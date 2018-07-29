import numpy as np
import time
import cv2
import math
import copy
import time

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)
    
def filter_region(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])       
    return cv2.bitwise_and(image, mask)

def select_region(image):
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.2, rows*0.55]
    top_left     = [cols*0.2, rows*0.5]
    bottom_right = [cols*0.6, rows*0.55]
    top_right    = [cols*0.6, rows*0.5]
   
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

def hough_lines(image):
    lines =  cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=15, minLineLength=5, maxLineGap=500)
    teste = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue
                slope = float(y2-y1)/float(x2-x1)
                angulo = math.degrees(math.atan(slope))
                if angulo < 0:
                    angulo = angulo * -1
                
                if(angulo < 15 or angulo > 180):
                    continue

                teste.append(line)
    return lines, teste

def main():
    cap = cv2.VideoCapture('H:\\python Linhas\\testevideo.avi')
    ret, frame = cap.read()
    f_todas = frame
    while(ret == True):
        start = time.time()
        f_todas = copy.copy(frame)
        gray         = convert_gray_scale(frame)
        edges        = detect_edges(gray)
        regions      = select_region(edges)
        todas, lines        = hough_lines(regions)

        if ret == True:
            if lines is not None: 
                for x in range(0, len(lines)):
                    for x1, y1, x2, y2 in lines[x]:
                            cv2.line(frame, (x1,y1), (x2,y2), (0,0,255),2)

            if todas is not None:
                for x in range(0, len(todas)):
                    for x1, y1, x2, y2 in todas[x]:
                            cv2.line(f_todas, (x1,y1), (x2,y2), (0,0,255),2)

            cv2.imshow('Linhas',f_todas)
            cv2.imshow('Região',regions)
            cv2.imshow('Video',frame)

        if cv2.waitKey(80) & 0xFF == ord('q'):
            break
        print(str(time.time() - start))
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

main()