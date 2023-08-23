import pathlib
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute()/"data/haarcascade_frontalface_default.xml"#haar cascade xml for detection

clf = cv2.CascadeClassifier(str(cascade_path))

img = cv2.imread("insangrubu.jpg")

while True:
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor =1.1,
        minNeighbors = 5,#the distance between two face to be detected
        minSize=(30,30),#minimum size for face to be detected
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for(x,y, width, height) in faces:
        cv2.rectangle(img, (x,y), (x+width , y+height), (255,255,0), 2)

    cv2.imshow("Faces", img)
    if cv2.waitKey(1)== ord("q"):
        break


cv2.destroyAllWindows()    