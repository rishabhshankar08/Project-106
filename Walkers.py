import cv2
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('walking.avi')
while(True):
    ret, frame = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(gray, 1.2, 3)
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 32:
        break
cap.release()
cv2.destroyAllWindows()