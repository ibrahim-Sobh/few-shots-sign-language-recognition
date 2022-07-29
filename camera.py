import cv2
import sys
import time
import os
from pygame import mixer 

video_path = sys.argv[1]
TIMER = 3
TIMER_START=0
TIMER_DURATION = 3

cap= cv2.VideoCapture(0)

width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer= cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, (width,height))
font = cv2.FONT_HERSHEY_SIMPLEX

mixer.init()
mixer.music.load('audios/beep-07a.wav', 'wav')


while True and TIMER_DURATION>0:
    ret,frame= cap.read()       
    if TIMER >0:
        frame=cv2.flip(frame,1)
        cv2.imshow('frame', frame)           

    k = cv2.waitKey(1)
    
    if k == ord('s'):
        prev = time.time()

        while TIMER >= 0:
            ret, frame = cap.read()
         
            text = str(TIMER)

            # get boundary of this text
            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            # get coords based on boundary
            textX = (frame.shape[1] - textsize[0]) / 2
            textY = (frame.shape[0] + textsize[1]) / 2
    
            frame=cv2.flip(frame,1)
            cv2.putText(frame, text, (int(textX), int(textY)),
                        font, 5, (0, 0, 255), 5, cv2.LINE_AA)
     
            cv2.imshow('frame', frame)
            
            cv2.waitKey(1)

            # current time
            cur = time.time()
            
            if cur-prev >= 1:
                mixer.music.play() 
                prev = cur
                TIMER = TIMER-1
        start_time = time.time()
                
    elif TIMER < 0:
        elapsed_time = time.time() - start_time
        
        if int(elapsed_time)>TIMER_DURATION:
            break
        
        text= "00:" + str(int(elapsed_time)).zfill(2)

        # get boundary of this text
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        
        textX = (frame.shape[1] - textsize[0])-50
        textY = (frame.shape[0] + textsize[1])-50
        
        frame=cv2.flip(frame,1)
        cv2.putText(frame, text, (int(textX),int(textY)),
        font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('frame', frame)        
        
        writer.write(frame)
        
        # Update and keep track of Countdownsq
        # if time elapsed is one second
        # than decrease the counter

    if k== ord('q'):
            break

cap.release()
writer.release()
cv2.destroyAllWindows()

if TIMER>0:
    sys.exit(0)
else:
    sys.exit(2)

