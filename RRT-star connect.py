import math
from pickle import FALSE, TRUE
import random
import cv2 as cv
import numpy as np



class Image:
    def __init__(self,img):
        self.img = img
        self.imgObstacle = None
        self.imgStart = None
        self.imgGoal = None
        self.contours_obstacle = None
        self.contours_start = None
        self.contours_goal = None
        self.start_node = None
        self.goal_node = None
        self.rows = img.shape[0]
        self.columns = img.shape[1]

    def imageHandling(self):
        imgHSV = cv.cvtColor(self.img,cv.COLOR_BGR2HSV)

        #Goal Image
        lower_green = np.array([45,100,20])
        upper_green = np.array([75,255,255])
        mask_start = cv.inRange(imgHSV,lower_green,upper_green)
        self.imgGoal = cv.bitwise_and(self.img,self.img,mask=mask_start)

        #Start Image
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask0 = cv.inRange(imgHSV, lower_red, upper_red)
        # upper mask (170-180)
        lower_red = np.array([170,50,50])
        upper_red = np.array([180,255,255])
        mask1 = cv.inRange(imgHSV, lower_red, upper_red)
        mask = mask0+mask1
        self.imgStart = cv.bitwise_and(self.img,self.img,mask=mask)

        #Black Border
        border = np.zeros((self.rows,self.columns), dtype=np.uint8)
        cv.rectangle(border,(7,7),(self.columns-7,self.rows-7), (255,255,255) ,thickness = -1)

        #Obstacle Image
        mask_obs = cv.bitwise_not(mask_start+mask)
        imgObstacle = cv.bitwise_and(self.img,self.img,mask=mask_obs)
        self.imgObstacle = cv.bitwise_and(imgObstacle,imgObstacle,mask=border)

    def getcontours(self,img):
        imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        imgBlur = cv.GaussianBlur(imgGray,(7,7),1)
        imgCanny = cv.Canny(imgBlur,50,50)
        imgDialated = cv.dilate(imgCanny,(7,7), iterations=3)
        contours, _ = cv.findContours(imgDialated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return contours

    def Centre(self,contour):
        c = contour
        M = cv.moments(c)
        centre_x = int(M["m10"] / M["m00"])
        centre_y = int(M["m01"] / M["m00"])
        return (centre_x,centre_y)
    
    def update(self):
        self.imageHandling()
        self.contours_obstacle = self.getcontours(self.imgObstacle)
        self.contours_start = self.getcontours(self.imgStart)
        self.contours_goal = self.getcontours(self.imgGoal)
        start_num = int(input("Enter start contour no.: "))
        goal_num = int(input("Enter goal contour no.: "))
        self.start_node = self.Centre(self.contours_start[start_num-1])
        self.goal_node = self.Centre(self.contours_goal[goal_num-1])


class Node:
    def __init__(self,coordinates,cost=None):
        self.coordinate = coordinates
        self.cost = cost
        self.childs = np.array([])
        self.parent = None

class connectedNode:
    def __init__(self,coordinates,parent_start,parent_goal,start_cost,goal_cost):
        self.coordinate = coordinates
        self.parentStart = parent_start
        self.parentGoal = parent_goal
        self.start_cost = start_cost
        self.goal_cost = goal_cost
        self.total_cost = start_cost + goal_cost


class RRTStarConnect:
    def __init__(self,start_node,goal_node,img,contours_obstacle,rows,columns):
        self.start_node = Node(start_node,0)
        self.goal_node = Node(goal_node,0)
        self.img = img
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        self.start = np.array([self.start_node])
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        self.goal = np.array([self.goal_node])
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        self.connected = np.array([])
        self.connected_count = 0
        self.rows = rows
        self.cols = columns
        self.radius = 25
        self.delta = 10
        self.contours_obs = contours_obstacle

    def pathplanning(self):
        c=0
        while TRUE:
            cv.imshow("Final",self.img)
            cv.waitKey(1)        
            if(c%2==0):
              flag = 0
            else:
              flag = 1
            randpt = self.RandomPt()
            nearNode,min_dist, = self.NearestNode(randpt,flag)
            x_new = self.Xnew(nearNode,randpt,min_dist)
            in_obstacle = self.CollisionDetection(x_new)
            if(in_obstacle==1):
              continue
            near_neighbours = self.NearNeighbours(x_new,flag)
            self.CostEstimate(near_neighbours,x_new,nearNode,min_dist,flag)
            self.linkXnew(x_new,flag)
            self.CanConnect(x_new,flag)
            if(self.connected_count >= 1200):
              self.plotPath()
              cv.destroyAllWindows()
              break
            self.Rewire(near_neighbours,x_new,flag)
            c=c+1


    def RandomPt(self):
        x = random.randint(0,self.cols)
        y = random.randint(0,self.rows)
        coordinate = [x,y]
        return tuple(coordinate)
    
    def NearestNode(self,randPt,flag):
        if(flag==0):
            min_dist = int(math.dist(randPt,self.start_node.coordinate))
            nearNode = self.start_node
            arr = self.start
        else:
            min_dist = int(math.dist(randPt,self.goal_node.coordinate))
            nearNode = self.goal_node
            arr = self.goal
        for node in arr:
            if(math.dist(randPt,node.coordinate) < min_dist):
                min_dist = int(math.dist(randPt,node.coordinate))
                nearNode = node
        return nearNode,min_dist        

    def Xnew(self,nearNode,randpt,min_dist):
        x_new = Node((0,0))
        if(min_dist <= self.delta):
            x_new.coordinate = (int(randpt[0]),int(randpt[1]))
        else:
            mag = pow(pow(randpt[0]-nearNode.coordinate[0],2)+pow(randpt[1]-nearNode.coordinate[1],2),0.5)
            x = int((((randpt[0]-nearNode.coordinate[0])/mag)*self.delta)+nearNode.coordinate[0])
            y = int((((randpt[1]-nearNode.coordinate[1])/mag)*self.delta)+nearNode.coordinate[1])
            x_new.coordinate = (x,y)
        return x_new

    def CollisionDetection(self,x_new):
        in_obstacle = 0
        for contour in self.contours_obs:
            dist = cv.pointPolygonTest(contour,x_new.coordinate,False)
            if(dist==1 or dist==0):
                in_obstacle = 1
                break
        return in_obstacle

    def CanConnect(self,x_new,flag):
        if(flag==0):
            for node in self.goal:
                if(int(math.dist(x_new.coordinate,node.coordinate))<=self.radius):
                    cost_goal = node.cost + int(math.dist(x_new.coordinate,node.coordinate))
                    connected_node = connectedNode(x_new.coordinate,x_new.parent,node,x_new.cost,cost_goal)
                    self.connected = np.append(self.connected,connected_node)
                    self.connected_count = self.connected_count + 1
                    break                    
        else:
            for node in self.start:
                if(int(math.dist(x_new.coordinate,node.coordinate))<=self.radius):
                    cost_start = node.cost + int(math.dist(x_new.coordinate,node.coordinate))
                    connected_node = connectedNode(x_new.coordinate,node,x_new.parent,cost_start,x_new.cost)
                    self.connected = np.append(self.connected,connected_node)
                    self.connected_count = self.connected_count + 1
                    break                  

    def NearNeighbours(self,xnew,flag):
        if(flag==0):
            arr = self.start
        else:
            arr = self.goal    
        neighbours = np.array([])
        for node in arr:
            if(math.dist(xnew.coordinate,node.coordinate) <= self.radius):
                np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
                neighbours = np.append(neighbours,node)
        return neighbours

    
    def CostEstimate(self,near_neighbours,x_new,x_near,min_dist,flag):   
        x_new.cost = x_near.cost + min_dist
        x_new.parent = x_near
        for node in near_neighbours:
            dist = int(math.dist(x_new.coordinate,node.coordinate))
            if(node.cost+dist<x_new.cost):
                x_new.parent = node
                x_new.cost = node.cost + dist
        
        x_new.parent.childs = np.append(x_new.parent.childs,x_new)
        if(flag==0):
            self.start = np.append(self.start,x_new)
        else:
            self.goal = np.append(self.goal,x_new)          


    def linkXnew(self,x_new,flag):
        if(flag==0):
            cv.line(self.img,x_new.coordinate,x_new.parent.coordinate,(255,0,0),thickness=1)
        else:
            cv.line(self.img,x_new.coordinate,x_new.parent.coordinate,(0,255,0),thickness=1)

    def Rewire(self,near_neighbours,x_new,flag):
        for node in near_neighbours:
            cost_neighbour = x_new.cost + int(math.dist(x_new.coordinate,node.coordinate))
            if(cost_neighbour<node.cost):
                childs = node.parent.childs
                for i in range(len(childs)):
                    if(childs[i]==node):
                        np.delete(childs, i)
                        break
                node.parent = x_new
                node.cost = cost_neighbour
                x_new.childs = np.append(x_new.childs,node)
                self.updateCost(node)
                self.linkXnew(node,flag)
             

    def updateCost(self,node):
        childs = node.childs
        for child_node in childs:
            child_node.cost = node.cost + int(math.dist(child_node.coordinate,node.coordinate))



    def plotPath(self):
        minCost = self.connected[0].total_cost
        bestNode = self.connected[0]
        for node in self.connected:
            if(node.total_cost<minCost):
                minCost = node.total_cost
                bestNode = node

        cv.line(self.img,bestNode.coordinate,bestNode.parentStart.coordinate,(0,0,255),thickness=2)
        cv.line(self.img,bestNode.coordinate,bestNode.parentGoal.coordinate,(0,0,255),thickness=2)
        
        node = bestNode.parentStart
        while(node != self.start_node):
            cv.line(self.img,node.coordinate,node.parent.coordinate,(0,0,255),thickness=2)
            node = node.parent
        node = bestNode.parentGoal
        while(node != self.goal_node):
            cv.line(self.img,node.coordinate,node.parent.coordinate,(0,0,255),thickness=2)
            node = node.parent 
        

img =cv.imread('task3.2.png')
myImage = Image(img)
myImage.update()



path = RRTStarConnect(myImage.start_node,myImage.goal_node,img,myImage.contours_obstacle,myImage.rows,myImage.columns)

path.pathplanning()

cv.imshow("Final2",img)
cv.waitKey(0)
cv.destroyAllWindows()


