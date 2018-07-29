#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
import time
import cv2
import math
    
# image is expected be in RGB color space
def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
###15
def apply_smoothing(image, kernel_size=3):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.2, rows*0.55]
    top_left     = [cols*0.2, rows*0.5]
    bottom_right = [cols*0.6, rows*0.55]
    top_right    = [cols*0.6, rows*0.5]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

def hough_lines(image):
    """
    `image` should be the output of a Canny transform.
    
    Returns hough lines (not the image with lines)
    """
    lines =  cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=0, maxLineGap=300)
    teste = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue # ignore a vertical line
                slope = float(y2-y1)/float(x2-x1)
                angulo = math.degrees(math.atan(slope))
                if angulo < 0:
                    angulo = angulo * -1
                print(angulo)
                
                if(angulo < 15 or angulo > 180):
                    continue

                teste.append(line)

    return teste

def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue # ignore a vertical line
            slope = float(y2-y1)/float(x2-x1)
            angulo = math.degrees(math.atan(slope))
            if angulo < 0:
                angulo = angulo * -1
            print(angulo)

            if(angulo < 70 or angulo > 150):
                continue

            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    
    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.6       # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    
    return left_line, right_line

    
def draw_lane_lines(image, lines, color=[0, 255, 0], thickness=12):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            p1, p2 = line
            cv2.line(line_image, p1, p2,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

def main():
    caminho = 'C:\\Users\\luanesteves\\Documents\\Projeto\\Placas\\'
    # ##imagem
    # image  = cv2.imread('C:\\Users\\luanesteves\\Documents\\Projeto\\2811.bmp', 0)
    # smooth_gray  = apply_smoothing(image)
    # edges        = detect_edges(smooth_gray)
    # regions      = select_region(edges)
    # # lines        = hough_lines(regions)
    # cv2.imshow('image',smooth_gray)
    # cv2.imshow('regions',regions)
    # cv2.imshow('edges',edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #video
    cap = cv2.VideoCapture('H:\\python Linhas\\testevideo.avi')
    ret, frame = cap.read()
    cont = 0
    while(ret == True):

        gray         = convert_gray_scale(frame)
        #smooth_gray  = apply_smoothing(gray)
        edges        = detect_edges(gray)
        regions      = select_region(edges)
        lines        = hough_lines(regions)

        if lines is not None:
            left_lane, right_lane = average_slope_intercept(lines)

        if ret == True:
            #cv2.imwrite(caminho + str(cont)+'.bmp', frame)
            cont = cont + 1
            if lines is not None: 
                for x in range(0, len(lines)):
                    for x1, y1, x2, y2 in lines[x]:
                            cv2.line(frame, (x1,y1), (x2,y2), (0,0,255),2)

            cv2.imshow('frame',frame)
            #cv2.imshow('smooth_gray',gray)
            cv2.imshow('regions',regions)
            print(cont)

        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
        
        ret, frame = cap.read()
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

main()