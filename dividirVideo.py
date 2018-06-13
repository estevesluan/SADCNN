import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import cv2

def main():

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('testevideo.avi',fourcc, 10.0, (640,480))
    
    #video
    cap = cv2.VideoCapture('H:\\python linhas\\output.avi')
    ret, frame = cap.read()
    cont = 0
    while(ret == True):
        if ret == True:
            if cont > 2440 and cont < 3000:
                #cv2.imwrite(caminho + str(cont)+'.bmp', frame)
                # write the flipped frame
                out.write(frame)
            cont = cont + 1
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
    # When everything done, release the capture
    cap.release()
    out.release()

main()