import cv2
import numpy as np

def closing(mask):
	kernel = np.ones((6,6), np.uint8)
	closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	return closed

def opening(mask):
	kernel = np.ones((6,6), np.uint8)
	opened = cv2.morpholgyEx(mask, cv2.MORPH_OPEN, kernel)
	return opened

def canny(mask):
	gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3,3), 0)
	edged = cv2.Canny(gray, 10, 250)
	return edged


lower_red = np.array([0, 90, 60], dtype=np.uint8)
upper_red = np.array([10, 255, 255], dtype=np.uint8)

lower_blue = np.array([90, 20, 60], dtype=np.uint8)
upper_blue = np.array([130,255,255], dtype=np.uint8)

red = [lower_red, upper_red, 'red']
blue = [lower_blue, upper_blue, 'blue']

colors = [red, blue]

def detectshape(image):
	edged = canny(image)
	mask = closing(edged)
	cnts, h = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	result = []
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02*peri, True)

		if len(approx) is 4:
			result.append(c)
	bw = np.zeros(image.shape[:2], dtype=np.uint8)*255

	for c in result:
		rect = cv2.minAreaRect(c)
		#box = cv2.cv.BoxPoints(rect)
		#box = np.int0(box)
		cv2.drawContours(bw, [c], -1,(255,255,255),-1)
		#cv2.rectangle(bw, box[0], box[1], box[2], box[3])

	image = cv2.bitwise_and(image, image, mask=bw)

	return image


def detectcolor(image):
	detectshape(image)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
	for color in colors:
		mask = cv2.inRange(hsv, color[0], color[1])
		cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		c = max(cnts, key=cv2.contourArea)
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02*peri, True)

		cv2.drawContours(image, [approx], -1, (0,255,0), 4)
	
	return image
	

myimage = cv2.imread('games.jpg')
ddimage = detectcolor(myimage)
cv2.imshow('test', ddimage)
cv2.waitKey(0)
