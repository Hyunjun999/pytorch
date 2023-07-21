import cv2

img = cv2.imread("./img.png")
img = cv2.resize(img, (500, 500))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
img_w_keypoints = cv2.drawKeypoints(img, keypoints, None)
cv2.imshow("Img w/ keypoints", img_w_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
