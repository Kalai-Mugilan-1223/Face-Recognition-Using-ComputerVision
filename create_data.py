import cv2
import os

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'  
sub_data = 'Kalai-Mugilan'     

path = os.path.join(datasets, sub_data)  
if not os.path.exists(path):
    os.makedirs(path)
(width, height) = (130, 100)   

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)

webcam = cv2.VideoCapture(0)  

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

count = 1
while count < 200:
    print(count)
    ret, frame = webcam.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(os.path.join(path, f"{count}.png"), face_resize)
    count += 1
    
    cv2.imshow('OpenCV', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
