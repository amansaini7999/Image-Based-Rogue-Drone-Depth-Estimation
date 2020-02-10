import cv2
import numpy as np
import time

path = "/home/aman/Documents/Project/DepthCal/Code/Data/data/left/"

yoloFiles = '/home/aman/Documents/Project/DepthCal/Code/Tiny Yolo Files/'

net = cv2.dnn.readNet(yoloFiles + 'yolov3-tiny.weights',yoloFiles + 'yolov3-tiny.cfg')
classes = []
with open(yoloFiles + 'coco.names','r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size = (len(classes),3))

dC = 50

img1 = cv2.imread(path + "1.jpg")
img1 = cv2.resize(img1, (480, 720))

img2 = cv2.imread(path + "3.jpg")
img2 = cv2.resize(img2, (480, 720))

def objectDetection(img):
    li = []
    w = 0
    height,width,channels = img.shape
    blob = cv2.dnn.blobFromImage(img,.00392,(416,416), (0,0,0),True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x0 = int(center_x - (w / 2))
                x1 = x0 + w

                y0 = int(center_y - (h / 2))
                y1 = y0 + h
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
                if x0 < 0:
                    x0 = 0
                if x1 > width:
                    x1 = width
                if y0 < 0:
                    y0 = 0
                if y1 > height:
                    y1 = height
    if w>0:
        li = [x0, y0, x1, y1]
        
    return li

def depth():
    li1 = objectDetection(img1)
    
    if(len(li1)>0):
        li2 = objectDetection(img2)

        pH_f1 = li1[3]-li1[1]
        pH_f2 = li2[3]-li2[1]

        aH = dC / (((1/pH_f1) - (1/pH_f2))*f)
        print(aH)

        dist = (aH*f)/pH_f2

        print(dist)

a = time.time()
f = 532
depth()

b = time.time()

print(1/(b-a))
