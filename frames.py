import cv2
import os
import sys
import time
import sys 

def save_frames(user_name,label, path_to_video, video_time=3):
    cap = cv2.VideoCapture(path_to_video)
    user = user_name
    path = "training/" + label + "/" + user 
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    capture_time = fps//(video_time**2)
    print(capture_time)
    count_frame = 0
    save_frame = 0

    while (True): 
        success, frame = cap.read()
        if frame is not None:
            count_frame = count_frame + 1
            if os.path.exists("videos/" + path):
                if count_frame >= capture_time:
                    cv2.imwrite(f'videos/{path}/images/{save_frame}.jpg', frame)
                    save_frame = save_frame + 1
                    count_frame = 0

            else:
                os.system("mkdir videos/training/"+label) 
                os.system("mkdir videos/training/"+label+"/"+user)                
                os.system("mkdir videos/training/"+label+"/"+user+"/images")
                cv2.imwrite(f'videos/{path}/images/{save_frame}.jpg', frame)
                save_frame = save_frame + 1
        else:
            break

    os.system("cp "+ path_to_video + " " + "videos/training/"+label+"/"+label+".mp4")
    cap.release()

    cv2.destroyAllWindows()
    print("Done")
def main():
    if len(sys.argv) > 0:
        path_to_video = sys.argv[1]
        label = sys.argv[2]
        user_name=sys.argv[3]
        save_frames(user_name,label, path_to_video)
        return 1
    else: 
        return 0
