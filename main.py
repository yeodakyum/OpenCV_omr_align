import cv2 as cv2
import numpy as np


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def order_point(ptru, ptrd):
    if (max(ptrd[0][1], ptrd[1][1]) > max(ptru[0][1], ptru[1][1])):
        if ptru[0][0] > ptru[1][0]:
            ptru[0], ptru[1] = ptru[1], ptru[0]
        if ptrd[0][0] < ptrd[1][0]:
            ptrd[0], ptrd[1] = ptrd[1], ptrd[0]
    else:
        if ptru[0][0] < ptru[1][0]:
            ptru[0], ptru[1] = ptru[1], ptru[0]
        if ptrd[0][0] > ptrd[1][0]:
            ptrd[0], ptrd[1] = ptrd[1], ptrd[0]
    return ptru + ptrd

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = np.float32(pts)
    print(rect)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped



img=cv2.imread('test.jpg') #read image
ih, iw, _ = img.shape
if ih > iw:
    img = cv2.resize(img, dsize=(1174, 1662), interpolation=cv2.INTER_AREA)

else:
    img = cv2.resize(img, dsize=(1662, 1174), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #turn image to gray
#cv2.imshow("test1",gray )
blur = cv2.GaussianBlur(gray,(5,5),0) #add blur
#cv2.imshow("test2",blur )
thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY_INV)[1]

#cv2.imshow("test",thresh )
#edges = cv2.Canny(blur,50,100) #find edges
edges = auto_canny(blur)
#cv2.imshow("3",edges )
contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find contours

ptrd = []
ptru = []
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if len(approx) == 4:
        #(x, y), radius = cv2.minEnclosingCircle(approx)
        (x, y, w, h) = cv2.boundingRect(approx)
        # draw the contour and center of the shape on the image

        center = (int(x), int(y))
        ar = w / float(h)

        mask = np.zeros(thresh.shape, np.uint8)
        mask = cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=3)
        mean = cv2.mean(img,mask)[:3]

        if int(w)< 30 and int(h)< 30 and (ar > 0.9 and ar < 1.1) and w*h>400 and sum(mean)<300:
            cv2.drawContours(img, [c], -1, (0, 0, 255), 2)
            ptru.append([x+w/2, y+h/2])

            print(mean)

        if int(w)< 25 and int(h)< 25 and (ar < 0.90 or ar > 1.1) and w*h>180 and sum(mean)<300:
            cv2.drawContours(img, [c], -1, (0, 0, 255), 2)
            ptrd.append([x+w/2, y+h/2])
            print(mean)


for [x,y] in ptrd:
    print(x,y)
    cv2.circle(img, (int(x),int(y)), radius=4, color=(255, 255, 255), thickness=-1)
for [x,y] in ptru:
    print(x,y)
    cv2.circle(img, (int(x),int(y)), radius=4, color=(255, 255, 255), thickness=-1)
resize = ResizeWithAspectRatio(img, width=640)
cv2.imshow('markers',resize) #show contours in green
ptr = ptru + ptrd
print(ptru, ptrd)
print(order_point(ptru, ptrd))
show = four_point_transform(img,order_point(ptru, ptrd))
resize = ResizeWithAspectRatio(show, width=640)
cv2.imshow("Result", resize )
cv2.waitKey(0)