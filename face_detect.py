import cv2 as cv

img = cv.imread('Photos/anh3.jpg')
# cv.imshow('Luke Shaw', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
eye_cascade = cv.CascadeClassifier('haar_eye.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

result = []
for (x, y, w, h) in faces_rect:
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
    if len(eyes) != 0:
        result.append([x, y, w, h])
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    # for (ex, ey, ew, eh) in eyes:
    #     cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
print(f'Number of faces found = {len(result)}')
cv.imshow('Detected Faces', img)

cv.waitKey(0)
