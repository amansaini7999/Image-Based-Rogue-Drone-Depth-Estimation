import cv2
import numpy as np
import time

imgF = cv2.imread("/home/aman/Documents/Project/DepthCal/Data/front.png", 0)

imgF = cv2.resize(imgF, (480, 720))

start = time.time()

imgF = imgF[300:338, 230:279]

liF = []
for i in range(imgF.shape[0]):
    for j in range(imgF.shape[1]):
        if imgF[i][j]>=180 and imgF[i][j]<=255:
            imgF[i][j] = 255
        else:
            imgF[i][j] = 0
            liF.append(i)
            
fy0 = min(liF)
fy1 = max(liF)

end = time.time()

print(1/(end-start))
