import cv2


trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_russian_plate_number.xml')
cam = cv2.VideoCapture('3.webm')


while cam.isOpened:
    succesful_frame_read,img = cam.read()
    grey_scale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grey_scale)
    print(face_coordinates)

        # a,b,c,d = face_coordinates[0]
    for x,y,w,h in face_coordinates:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)

        this_plate = img[y:y+h,x:x+w]

        cv2.imwrite("1.jpg",this_plate)


    cv2.imshow("face Detection",img)
    key = cv2.waitKey(1)

print("code completed")