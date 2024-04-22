# import required libraries
import cv2
import numpy as np
import math
import random

# input the image
img = cv2.imread('map.jpeg')
# convert the image to greyscale for creating binary image
blur = cv2.medianBlur(img, 15)
gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(gray, 244, 255, cv2.THRESH_BINARY)


def distance(coord1, coord2):
    distance = math.sqrt(pow((coord1[0]-coord2[0]),2) + pow((coord1[1]-coord2[1]),2))
    return distance

def checkObstacle(coord1, coord2):
    if coord1[0] != coord2[0]:
        slope = (coord2[1]-coord1[1])/(coord2[0]-coord1[0])
        intercept = (coord1[1]*coord2[0] - coord2[1]*coord1[0])/(coord2[0]-coord1[0])
        
        if abs(coord1[0]-coord2[0]) > abs(coord1[1]-coord2[1]):   
            if coord1[0]>coord2[0]:
                temp = coord1
                coord1 = coord2
                coord2 = temp
                
            for i in range(coord1[0], coord2[0]):
                j = slope*i + intercept
                if thresh[math.floor(j),i] == 0:
                    return True
       
        else:
            if coord1[1]>coord2[1]:
                temp = coord1
                coord1 = coord2
                coord2 = temp
                
            for j in range(coord1[1], coord2[1]-1):
                i = (j-intercept)/slope
                if thresh[j,math.floor(i)] == 0:
                    return True

    if coord1[0] == coord2[0]:
        if coord1[1]>coord2[1]:
            temp = coord1
            coord1 = coord2
            coord2 = temp
        i = coord1[0]
        for j in range(coord1[1], coord2[1]-1):
            if thresh[j,i] == 0:
                return True
            elif (i,j) == coord2:
                return False  
         
    return False
            
        
# Change the initial and final coordinates here
x1, y1, x2, y2 = 677, 1056, 2161, 1159

pStart = (x1,y1)
pFinal = (x2,y2)

plotCoords = [pStart]
whitePoints = []
adjList = {pStart: []}


width = np.shape(thresh)[0]
height = np.shape(thresh)[1]

colImage = np.zeros((width, height,3), dtype="uint8")

colImage[:,:,0] = thresh # for red
colImage[:,:,1] = thresh # for green
colImage[:,:,2] = thresh # for blue

# blur[y1, x1] = (128, 0, 128)
cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)
# blur[y2, x2] = (128, 0, 128)
cv2.circle(img, (x2, y2), 10, (255, 0, 0), -1)
count = 0

for i in range(height):
    for j in range(width):
        if thresh[j,i] == 255:
            whitePoints.append((i,j))


thresholdDist = 60 

while True:

    randI = random.randint(0, len(whitePoints))
    leastD = 3200
    
    #Step-1: Check for Obstacles & Find Closest Point 
    for point in plotCoords: 
        if not checkObstacle(point,whitePoints[randI]):
            dist = distance(point,whitePoints[randI])
            if dist < leastD:
                leastD = dist
                nearPoint = point   
    
    #Step-2: Store the child node along the Closest Point and Random Point 
    if leastD < 3200: #nearPoint has been found
        if (nearPoint[1] - whitePoints[randI][1] != 0):
            sign = -(nearPoint[1] - whitePoints[randI][1]) / abs(nearPoint[1] - whitePoints[randI][1])
        elif (nearPoint[0] - whitePoints[randI][0] != 0):
            sign = -(nearPoint[0] - whitePoints[randI][0]) / abs(nearPoint[0] - whitePoints[randI][0])
        else:
            sign = 0

        print(sign)
            
        if (nearPoint[0]-whitePoints[randI][0]) != 0:
            newSlope = (nearPoint[1]-whitePoints[randI][1]) / (nearPoint[0]-whitePoints[randI][0])

            theta = math.atan(newSlope)
            if theta < 0: theta1 = np.pi + theta
            else: theta1 = theta
            finalPoint = (nearPoint[0]+int(sign*thresholdDist*math.cos(theta1)), int(nearPoint[1]+int(sign*thresholdDist*math.sin(theta1))))
            
        if (nearPoint[0]-whitePoints[randI][0]) == 0:
            finalPoint = (nearPoint[0], int(nearPoint[1] + sign*thresholdDist))
            
        if (finalPoint[0] >= height or finalPoint[1] >= width): finalPoint = whitePoints[randI]
        elif (checkObstacle(nearPoint, finalPoint)): finalPoint = whitePoints[randI]

        plotCoords.append(finalPoint)
        
        adjList[nearPoint].append(finalPoint)
        
        if finalPoint in adjList:
            adjList[finalPoint].append(nearPoint)
        else:
            adjList[finalPoint] = [nearPoint]
        
        #print(newSlope)
        print(nearPoint)
        print(int(thresholdDist*math.cos(theta)), int(thresholdDist*math.sin(theta)))
        print(finalPoint)

        
        cv2.line(img, nearPoint, finalPoint, (0,0,255), 3)

        if (distance(pFinal, finalPoint) < 2 * thresholdDist  and not checkObstacle(pFinal, finalPoint)):
            cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
            cv2.line(img, pFinal, finalPoint, (0,255, 0), 5)
            cv2.imshow('thresh', img)
            print("Done")

            plotCoords.append(pFinal)
            
            adjList[finalPoint].append(pFinal)
            
            adjList[pFinal] = [finalPoint]

            point = pFinal
            while point != pStart:
                cv2.line(img, point, adjList[point][0], (0,255,0), 5)
                point = adjList[point][0]

            cv2.destroyAllWindows()
            print("displayed")
            cv2.imshow('thresh', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

            
        
        cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
        
        if (not count % 10): 
            cv2.imshow('thresh', img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        
                   
    count+=1

print("Finished")