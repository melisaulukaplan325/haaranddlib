import dlib
import cv2
#pretrained data set for landmarks
p = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
#face predictor from dlib library
face_predictor = dlib.shape_predictor(p)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector(gray)

    for face_rect in detected_faces:
        
        left = face_rect.left()
        top = face_rect.top()
        right = face_rect.right()
        bottom = face_rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        #finding landmarks to frame
        shape = face_predictor(gray, face_rect)
       #placing that landmarks to the face between (0,68)
        for n in range(0, 68):
            x = shape.part(n).x
            y = shape.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

       
    
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
