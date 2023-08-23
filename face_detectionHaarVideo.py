import cv2
import pathlib

capture = cv2.VideoCapture(0)
cascade_path = pathlib.Path(cv2.__file__).parent.absolute()/"data/haarcascade_frontalface_default.xml"#for finding the way to haar cascade xml
classifier = cv2.CascadeClassifier(str(cascade_path))

while True:
   ret,frame = capture.read()
   if not ret:#frame gets controlled
        break

   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#conversion to gray scale
   faces = classifier.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags= cv2.CASCADE_SCALE_IMAGE)#minBeighbor means the closest face that can be detcted
   for(x,y,w,h) in faces:
      cv2.rectangle(frame, (x,y),(x+w,y+h),(255,255,0),2)#rectangle to box the face

   cv2.imshow("Video", frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
  

