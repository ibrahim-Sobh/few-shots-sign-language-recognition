
import cv2
import tensorflow as tf
import mediapipe as mp
import pickle
import numpy as np
import sys
import os

sys.path.append("..")
video_path = sys.argv[1]
file_code = sys.argv[2]
path =0 if  video_path =="0" else video_path 
cap= cv2.VideoCapture(path)
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter("videos/inference/"+file_code+".mp4", fourcc, 30, (width,height))
model = tf.keras.models.load_model("sign/train_tl")

with open('sign/labels_encoder.pkl', 'rb') as f:
    labels_encoder = pickle.load(f)

keypoints_frames = []
count = 0
result_label = ""


le = labels_encoder

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def result(path,result_label,file_code):
# calculate stuff
    if result_label is None or result_label=="":
        return 0
    with open('temp/result.txt', 'w') as fh:
        fh.write(result_label[0])
    return result_label[0]
    

def extract_keypoints(results):
    hand_label_result = dict()
    for hand in results.multi_handedness:
        hand_label_result[hand.classification[0].index] = hand.classification[0].label

    keypoints = []
    for hand in results.multi_hand_landmarks:
        hand_keypoints = []
        for point in range(21):
            landmark = hand.landmark[point]
            hand_keypoints.extend([landmark.x, landmark.y, landmark.z])
            
        keypoints.append(hand_keypoints)
        
    #print(len(keypoints))
    #print(hand_label_result)
    
    if len(keypoints) == 1:
        keypoints.append(list(np.zeros(len(keypoints[0]))))
        
    elif len(keypoints) == 0:
        keypoints.append(list(np.zeros(21*3)))
        keypoints.append(list(np.zeros(21*3)))
        
    try:
        if hand_label_result[0] != 'Left':
            keypoints = keypoints[1] + keypoints[0]
        else:
            keypoints = keypoints[0] + keypoints[1]
    except:
        keypoints = keypoints[1] + keypoints[0]
    return keypoints

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
  while cap.isOpened():
    success, image = cap.read()
    if image is None:
        break
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    count += 1
    if results.multi_hand_landmarks:
      key_points = extract_keypoints(results)
      keypoints_frames.append(key_points)
      if np.array([keypoints_frames]).shape[1]>=43 and count >=43:
        predict = model.predict(np.array([keypoints_frames]))[0]
        max_label = np.argmax(predict)
        if predict[max_label] > 0.5:
            result_label = le.inverse_transform([max_label])
        else:
            result_label = ""
        count = 0
        keypoints_frames = []
    
      for hand_landmarks in results.multi_hand_landmarks:
        #print('hand_landmarks:', hand_landmarks)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        #print(hand_landmarks)
        #print(mp_hands.HAND_CONNECTIONS)
        
    # Flip the image horizontally for a selfie-view display.
    cv2.putText(image, ' '.join(result_label), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('MediaPipe Hands', image)
    output.write(image)
    
    k = cv2.waitKey(1)
    if k ==ord('q'):
        break
    if k ==ord('c'):
        sys.exit()
result(path,result_label,file_code)
output.release()
cap.release()

