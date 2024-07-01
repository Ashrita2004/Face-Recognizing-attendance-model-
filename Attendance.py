import csv
import face_recognition
import numpy as np
import cv2 as cv
import os
from datetime import datetime

video = cv.VideoCapture(0)

def load_encodings(imgs_path):      
    student_encodings = []    
    student_names = []        

    for filename in os.listdir(imgs_path):
        img_path = os.path.join(imgs_path, filename)
        student_img = face_recognition.load_image_file(img_path)
        student_encoding = face_recognition.face_encodings(student_img)[0]  # As face_encodings returns a list of numerical representation of img, first one has been accepted
        student_encodings.append(student_encoding)
        student_names.append(os.path.splitext(filename)[0]) 
    return student_encodings, student_names

imgs_path = 'E:\coding\python\students_img'

student_encodings, student_names = load_encodings(imgs_path)

#video = cv.VideoCapture(0)
video = cv.VideoCapture('Video_File.mp4')
students = student_names.copy()
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
face_names = []
while True:
    ret, frame = video.read()
    resized_frame = cv.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_resized = resized_frame[:, :, ::-1]                                       
    face_locations = face_recognition.face_locations(rgb_resized)                 
    face_encodings = face_recognition.face_encodings(rgb_resized,face_locations) 
    face_names = []
    for face_encoding, face_location in zip(face_encodings,face_locations):
        matching = face_recognition.compare_faces(student_encodings,face_encoding)
        similarities = face_recognition.face_distance(student_encodings,face_encoding)  
        best_match_index = np.argmin(similarities)                                      
        if matching[best_match_index]:
            name = student_names[best_match_index]
        else:
            name = "STUDENT ENTRY NOT FOUND" 

        face_names.append(name)
        if name in student_names:
            if name in students:
                students.remove(name)  #to avoid multiple names of same student
                print(students)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name,current_time])

        top, right, bottom, left = face_location
        top *= 4                        
        right *= 4
        bottom *= 4
        left *= 4
        
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 5)           

        cv.rectangle(frame, (left, bottom-40), (right, bottom), (0, 0, 0), cv.FILLED)        
        cv.putText(frame, name, (left + 6, bottom - 5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 1)

    cv.imshow('Attendance system', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()    
