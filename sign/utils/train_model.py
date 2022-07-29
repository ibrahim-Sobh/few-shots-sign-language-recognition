import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.utils import shuffle
import glob
import mediapipe as mp
import cv2
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

PATH = "sign/"

def augment(image_label):
    image = image_label
    thr = tf.random.uniform([1])/4
    sign = tf.random.uniform([1], maxval = 2, dtype=tf.dtypes.int32)*2-1
    image = image + tf.cast(sign, dtype=tf.dtypes.float32, name=None)*thr

    shift = tf.random.uniform(shape=[1], minval=-5, maxval=5, dtype=tf.int32)
    image = tf.roll(image, shift=shift, axis=[0])

    return image

def save_labels_keypoints(labels, keypoints):
    with open(PATH + 'labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open(PATH + 'keypoints.pkl', 'wb') as f:
        pickle.dump(keypoints, f)

def train_process(new_label:str, new_keypoints:list):

    le = preprocessing.LabelEncoder()

    try:
        with open(PATH + 'keypoints.pkl', 'rb') as f:
            keypoints = pickle.load(f)
            keypoints.append(new_keypoints[0])
    except:
        keypoints =  new_keypoints
    
    try:
        with open(PATH + 'labels.pkl', 'rb') as f:
            labels = pickle.load(f)
            labels.extend([new_label])
    except:
        labels = [new_label]

    save_labels_keypoints(labels, keypoints)

    le.fit(labels)

    with open(PATH + 'labels_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    labels_num = le.transform(labels)
    y = to_categorical(labels_num).astype(int)

    
    keypoints = np.array(keypoints)
    #print(keypoints)
  
    keypoints_pad = tf.keras.preprocessing.sequence.pad_sequences(keypoints, maxlen=43, dtype='float32',)

    X_train, y_train = shuffle(keypoints_pad, y, random_state=0)

    batch_size = 128

    tf_train_data = tf.data.Dataset.from_tensor_slices((np.array([X_train]) ,
                                                        np.array([y_train]))).shuffle(batch_size, 
                                                                                    reshuffle_each_iteration=True)

    tf_train_data = (tf_train_data.shuffle(batch_size * 100)
        .map(lambda x, y: (augment(x), y),
        num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))

    model = Sequential()
    model.add(Bidirectional(LSTM(30, return_sequences=True, activation='relu', input_shape=(None,126))))
    model.add(LSTM(30, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(le.classes_.shape[0], activation='softmax'))

    op = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(tf_train_data, epochs=200, batch_size=batch_size)
    model.save(PATH + "train_tl")


def dataset_path_labels(path_dataset):
    
    txtfiles = []
    labels = []
    files = []
    names = []
    for file in glob.glob(path_dataset + "/*"):
        label = file.split("/")[-1]
        
        for file_img in glob.glob(file + "/*/images/*.jpg"):
            name = file_img.split("/")[3]
            
            if names == []:
                txtfiles.append(files)
                names.append(name)
                labels.append(label)
                
            if name in names:
                files.append(file_img)
                
            else:
                txtfiles.append(files)
                files = []
                names.append(name)
                labels.append(label)
                
    return txtfiles, labels


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
        
    if len(keypoints) == 1 :
        keypoints.append((np.zeros(len(keypoints[0]))).tolist())
    #elif len(keypoints) == 0 or keypoints == [[]]:
    #    keypoints.append((np.zeros(21*3*2)).tolist())

    try:
        if hand_label_result[0] != 'Left':
            keypoints = keypoints[1] + keypoints[0]
        else:
            keypoints = keypoints[0] + keypoints[1]
    except:
        keypoints = keypoints[1] + keypoints[0]
    
    #if len(keypoints) == 21*3:
    #    keypoints.extend((np.zeros(63)).tolist())
    if keypoints == []:
        print("keypoints is empty")
    return keypoints


def collect_keypoints(txtfiles):

    keypoints = []

    # For static images:
    IMAGE_FILES = txtfiles

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.45) as hands:
        
    
        for n, file_label in enumerate(IMAGE_FILES):

            keypoints_label = []
            for idx, file in enumerate(file_label):

                image = cv2.flip(cv2.imread(file), 1)

                image = cv2.imread(file)
                # Convert the BGR image to RGB before processing.
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.multi_hand_landmarks:
                    keypoints_label.append(list(np.zeros(21*3*2)))
                    continue
                keypoints_label.append(extract_keypoints(results))
             
                image_height, image_width, _ = image.shape
                annotated_image = image.copy()

                for hand_landmarks in results.multi_hand_landmarks:

                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
         
            keypoints.append(np.array(keypoints_label))
    
    return keypoints

def new_keypoints_labels(path_dataset):
    txtfiles, labels = dataset_path_labels(path_dataset)
    keypoints = collect_keypoints(txtfiles)
    save_labels_keypoints(labels, keypoints)


def train_new_data(new_label:str, user:str):
    print("Training new data...")
    path ="videos/training/{}/{}/".format(new_label,user)
    all_paths_images= []
    for (dirpath, dirnames, filenames) in os.walk(path):
        all_paths_images.extend(filenames)
    all_paths_images=[os.path.join(path,"images", x) for x in all_paths_images]
    all_paths_images.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    new_keypoints = collect_keypoints([all_paths_images])
    train_process(new_label, new_keypoints)
