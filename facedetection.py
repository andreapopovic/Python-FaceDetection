import cv2 as cv
import mediapipe as mp
import numpy as np
from tkinter import *
from tkinter.ttk import *
import tensorflow

capture = cv.VideoCapture(0) #to open Camera

numOfDetectedFaces=0

size = 45
def faceDetection(detectionType):
    text="Algorithm: "
    if detectionType==1:
        text+="Viola Jones detection"
        ViolaJonesDetection(text)
    if detectionType==3:
        text+="Face mash detection"
        FaceMashDetection(text)
  

def ViolaJonesDetection(algorithm):
    pretrained_model = cv.CascadeClassifier("face_detector.xml") 
    numOfDetection=60
    textForFile="Human face is detected with Viola Jones Detection on positions: "
    while True:
        boolean, frame = capture.read()
        cv.putText(frame,algorithm,(15,25),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),2,cv.LINE_AA)
        if boolean == True:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            coordinate_list = pretrained_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) 
            numOfDetectedFaces= len(coordinate_list)
            cv.putText(frame,"Number of detected faces: " + str(numOfDetectedFaces),(15,40),cv.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2,cv.LINE_AA)
            for (x,y,w,h) in coordinate_list:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                coordinates="x:"+str(x)+" y:"+str(y)
                if numOfDetection==60:
                    with open('coordinates.txt','w') as f:
                        f.write(textForFile)
                        numOfDetectedFaces=0
                else:
                    numOfDetection+=1
            displayVideoFrame(frame)
            if cv.waitKey(5) & 0xFF == 27:
                break  
    capture.release()
    cv.destroyAllWindows()


def FaceMashDetection(algorithm):
    numOfDetection=60
    textForFile="Human face is detected with Face Mash Detection on positions: "
    # Face mesh detection
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while capture.isOpened():
            success, image = capture.read()
            image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
            cv.putText(image,algorithm,(15,25),cv.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),2,cv.LINE_AA)
            image.flags.writeable = False
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                numOfDetectedFaces=len(results.multi_face_landmarks)
                cv.putText(image,"Number of detected faces: " + str(numOfDetectedFaces),(15,40),cv.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2,cv.LINE_AA)
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                    (height, width, channel) = image.shape
                    coordinates="x:"+str((results.multi_face_landmarks[0].landmark[4].x*width))+" y:"+str(results.multi_face_landmarks[0].landmark[4].y*height)
                    if numOfDetection==60:
                        with open('coordinates.txt','w') as f:
                            f.write(textForFile+coordinates)
                            numOfDetectedFaces=0
                    else:
                        numOfDetection+=1

            else:
                cv.putText(image,"Number of detected faces: " + str(0),(15,40),cv.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2,cv.LINE_AA)           
            displayVideoFrame(image)
            if cv.waitKey(5) & 0xFF == 27:
                break
    capture.release()
    cv.destroyAllWindows()

def displayVideoFrame(frame):
   # size= 45
    (height, width, channel) = frame.shape 
    button = cv.imread("buttonChangeAlgorithm.png")
    button= cv.resize(button, (size*2,size))
    button2Gray = cv.cvtColor(button,cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(button2Gray,1,255,cv.THRESH_BINARY)
    roi = frame[10:10+size,width-size*2-10:width-10]
    roi[np.where(mask)] = 0
    roi+= button
    cv.namedWindow("LiveFaceDetection",cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("LiveFaceDetection",cv.WND_PROP_FULLSCREEN,0)
    cv.imshow("LiveFaceDetection", frame)
    cv.setMouseCallback("LiveFaceDetection",click_event)

def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        boolean, frame = capture.read()
        (height, width, channel) = frame.shape
       # print("Width:"+str(width)+"Height:"+str(height)+"Size:"+str(size))
        if (x >= (width-size*2-10) & x <= (width-10) ) :
           
          # print("Tu ste: x="+str(x)+" y="+ str(y) )
           master = Tk()
           master.geometry("200x200")
           master.title("Select an algorithm")
           label = Label(master,text="Select an algorithm")
           label.pack(pady=10)
           btn1 = Button(master,text="ViolaJonesDetection",command= lambda:[master.destroy(),faceDetection(1)]).pack()
           btn2 = Button(master,text="FaceMashDetection",command= lambda:[ master.destroy(),faceDetection(3)]).pack()
           btn3 = Button(master,text="Quit",command= master.destroy).pack()
           mainloop()
       
        
faceDetection(3)

