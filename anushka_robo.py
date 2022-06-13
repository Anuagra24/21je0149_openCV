
import cv2
import numpy as np
import imutils
code_color = [[175],[147],[222],[0]]


marker_path = [

    "C:\\Users\\Anushka\\roboProjects\\resource\Ha.jpg",
    "C:\\Users\\Anushka\\roboProjects\\resource\\HaHa.jpg",
    "C:\\Users\\Anushka\\roboProjects\\resource\XD.jpg",
    "C:\\Users\\Anushka\\roboProjects\\resource\LMAO.jpg"
]
img_path = "C:\\Users\\Anushka\\roboProjects\\resource\\CVtask.jpg"


image = cv2.imread(img_path)

image = cv2.resize(image,None,None,fx=0.25,fy=0.25)
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#defining a function to read all the ids of markers
def findAurco(img,marker_size=5,total_marker=250,draw=True):
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key=getattr(cv2.aruco,f'DICT_{marker_size}X{marker_size}_{total_marker}')
    arucoDict=cv2.aruco.Dictionary_get(key)
    arucoParem=cv2.aruco.DetectorParameters_create()
    (bbox,ids,_)=cv2.aruco.detectMarkers(grey,arucoDict,parameters=arucoParem)
    arucoID = ids[0][0]
    print(arucoID)

for marker in marker_path:
    marker_name = cv2.imread(marker)
    print('ID OF ',marker)
    findAurco(marker_name)





image_grey = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
imgResult = image.copy()
thresh, thresh_img = cv2.threshold(image_grey,223,255,cv2.THRESH_BINARY)

cont = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)

cont_img = image.copy()

for r in code_color:
    image = cv2.medianBlur(image, 3, None)

    lower_bound = np.array(r)
    upper_bound = np.array(r)
    mask = cv2.inRange(image_grey,lower_bound,upper_bound)
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    array_box = []
    for c in cont:
        if cv2.contourArea(c)<5000:
            continue
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
        if len(approx) == 4:
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype='int')
            boxs = box.copy()
            boxs= boxs.tolist()
            array_box.append(boxs)
            cv2.drawContours(cont_img,[c],-1,(0,255,0),2)
            cv2.drawContours(cont_img,[box],-1,(0,255,0),1)

array_box.remove([[0, 308], [0, 0], [436, 0], [436, 308]])
for i in range(0,4):
    (tl, tr, br, bl) = array_box[i]
    marker = cv2.imread(marker_path[i])
    h, w, c = marker.shape
    pts1 = np.array((tl, tr, br, bl))
    pts2 = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(marker, matrix, (image.shape[1], image.shape[0]))
    imgResult = cv2.bitwise_or(imgResult, imgOut)


cv2.imshow('final.jpg',imgResult)
cv2.waitKey(0)
