import sys
sys.path.append("..")
import time
import streamlit as st
import subprocess
import os
from os.path import exists
import uuid 
from PIL import Image
from frames import save_frames
import sign.utils.train_model as train_model


st.set_page_config(layout="wide")

### Styling the App
streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css?family=Open+Sans&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Open Sans'
			}
			</style>
			"""
   
st.markdown(streamlit_style, unsafe_allow_html=True)
# Live Camera Stream


# functions for recording detection
def sign_recognition_video():
    new_title = '<p style="font-size: 42px; font-weight:bolder;">Sign Language Recognition for &#127909;<br/></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown("""<p style="font-size: 25px; font-weight:bolder;">
    This Sign Language Detection Model takes in a video as an input and then outputs that video """ +
    """along side the sign word equivalant in natutral Langauge</br></br> &#129330; &#10133; &#129302; &#10145;"""+
    """ Thank you !</p></br>""", unsafe_allow_html=True)
    
    st.subheader("Option 1 - Live Beta")
    st.markdown("<p style='font-size: 20px' ><b>Instructions</b><ul><li>Press ( C ) to Close</p>", unsafe_allow_html=True)
    run_live =st.button("Launch Live Webcam Recognizer")
    if run_live:
        process =subprocess.run(["python", "sign/utils/test.py","0","live"])
        
    st.subheader("Option 2 - Upload a video")
    file = st.file_uploader('', type = ['mp4'])
  
    if file is not None:
        file_path= os.path.join("videos/keepers",file.name)
        st.video(file)
        run_recoginzer =st.button("Run Recognizer")
        if run_recoginzer: 
            with open(file_path,"wb") as f: 
                f.write(file.getbuffer())         
                status_results=st.info("Running File...")
                process =subprocess.run(["python", "sign/utils/test.py",os.path.join("videos/keepers",file.name),file.name.replace(".mp4","")])
                time.sleep(2)
                status_results.empty()
                st.success("Results Ready...")
                st.video("videos/inference/"+file.name)   
                f = open("temp/result.txt","r")
                x = (f.read())
                st.success(x)
                
    st.markdown("<hr style= size='6', color=black> ", unsafe_allow_html=True)
    st.subheader("Option 3 - Redcord using Webcam")
    st.markdown("<p style='font-size: 20px' ><b>Instructions</b><ul><li>Press ( S ) to Start</li><li>Wait the "+
                "timer for 3 Seconds</li><li>Press ( Q ) to Quit</p> ", unsafe_allow_html=True)
    
    
    run =st.button("Launch Recording Webcam")

    if run:
        file_code =str(uuid.uuid4())[:8]
        path ="videos/keepers/"+file_code+".mp4"
        process =subprocess.run(["python", "camera.py",path])
       
        print(path)
        if process.returncode ==0:
            st.write("Nothing Was Redorded!")
        
        elif exists(path):
            st.write("Video Was Redorded!")
            st.video(path)
            status_results=st.info("Running File...")
            process =subprocess.run(["python", "sign/utils/test.py",'videos/keepers/'+file_code+'.mp4',file_code])
            time.sleep(2)
            status_results.empty()
            st.success("Results Ready...")
            f = open("temp/result.txt","r")
            x = (f.read())
            st.video("videos/inference/"+file_code+".mp4")
            st.info(x)
            keep =st.checkbox("Keep Video")
            delete=st.checkbox("Delete Video")
            if keep:
                st.write("Video Was Submited Sucessefully!!")
                return file_code
            elif delete:
                 pass

def sign_recognition_Video_retraining():
    new_title = '<p style="font-size: 42px; font-weight:bolder;">Teach me a Sign &#129305;<br/></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown("""<p style="font-size: 25px; font-weight:bolder;">
    This Sign Language Detection Model can learn a new Sign Langauge gesture only from taking a video as an input from the user """ +
    """</br></br> &#127909; &#10133; &#129302; &#10145;"""+
    """ My New Sign &#128077;!</p></br>""", unsafe_allow_html=True)
  
    st.subheader("Redcord using Webcam")
    st.markdown("<p style='font-size: 20px' ><b>Instructions</b><ul><li>Press ( S ) to Start</li><li>Wait the "+
                "timer for 3 Seconds</li><li>Press ( Q ) to Quit</p> ", unsafe_allow_html=True)
    
    col1, col2, col3= st.columns(3)
    with col1:
        title = st.text_input('Give your Sign a Name', 'Name goes here')
        
    run =st.button("Launch Webcam")
    file_code =title.strip()
    path ="videos/keepers/"+file_code+".mp4"
    
    if run:
        if len(title) ==0 :
            st.error("Please Enter a Sign Name")
        elif title == "Name goes here":
            st.error("Please Enter a valid Sign Name")
        elif len(title) > 15 :
            st.error("You can't use more than 10 characters")
        else:
            process_training =subprocess.run(["python", "camera.py",path])
            print(path)
            if process_training.returncode ==0:
                st.write("Nothing Was Redorded!")
            
            elif exists(path):
                st.write("Video Was Redorded!")
                st.video(path)
            
                st.write('Name your Sign Language', title)
                #process_training =subprocess.run(["python", "frames).py",'videos/keepers/'+file_code+'.mp4',title])
                save_frames(os.environ['USER'],title,'videos/keepers/'+file_code+'.mp4')
                train_model.train_new_data(title,os.environ['USER'])
                keep =st.checkbox("Keep Video")
                delete=st.checkbox("Delete Video")
                if keep:
                    st.write("Video Was Submited Sucessefully!!")
                    return file_code
                elif delete:
                    pass
    
def main():
    new_title = '<p style="font-size: 42px; font-weight:bolder;">SignMe &#128406;<br/></p><p style="font-size: 32px;">Welcome to our App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""<p style="font-size: 24px;text-align: left;">
    This research project is built to demonstrate the importance of <b>Few Shot Learning in Sign Language Recognition</b>.
    It was built using Streamlit and Mediapipe and many more computer vision Libraries.""",unsafe_allow_html=True)
    img_0=st.image("images/signme.png", width=660)
    line_0=st.markdown("<hr style= size='6', color=black> ", unsafe_allow_html=True)
    img_1=st.image("images/github.png", width=50)
    read_repo = st.markdown("""Our Github repository can be found 
    [here](https://github.com/Isaacgv/action-learning/tree/main)""")
    
    st.sidebar.title("Select Activity")
    
    choice  = st.sidebar.selectbox("MODE",("About","Instructions","Sign Language Recognition", "Teach me a Sign")) # "Sign Language Recognition(Image)",
    #["Show Instruction","Landmark identification","Show the #source code", "About"]
    if choice == "About":
        print()
        
    elif choice == "Sign Language Recognition":
        img_0.empty()   
        img_1.empty()  
        line_0.empty() 
        read_me_0.empty()
        read_me.empty()
        read_repo.empty()
        
        sign_recognition_video()

        
    elif choice == "Teach me a Sign":
        img_0.empty()   
        img_1.empty()  
        line_0.empty() 
        read_me_0.empty()
        read_me.empty()
        read_repo.empty()
   
        sign_recognition_Video_retraining()

    elif choice=="Instructions":
        img_0.empty()   
        img_1.empty()  
        line_0.empty() 
        read_me_0.empty()
        read_me.empty()
        read_repo.empty()
        st.markdown(""" ## User Guide: :clipboard:

        * Using our application you will be able to detect, translate sign language into natural words
        * Create/use your own sign language too ! 
        * This application is for educational purposes and anyone who wants to discover and learn
        sign language

        P.S We are using American Sign Language (ASL) you can find more details here
        [ASL](https://www.signingsavvy.com/)

        ## Features:
        ##### Detecting Sign Language :
        There are two ways you can detect sign language:
        * Click on Sign Language Recognition(Video)""")
        
        st.markdown("##### Option 1")
        st.markdown(""" 
        * You can open your Webcam on our website and get Live Feedback
        * Click on the Launch Live Webcam button to open the webcam
        *  Press on C to Close When you are done """)
        image_0 = Image.open("images/live.png")
        st.image(image_0)

        st.markdown(""" ##### Option 2
        * Upload videos from your file 
        """)
        image = Image.open("images/upload.png")
        st.image(image)

        st.markdown(""" * After upload a video you can replay, save it or use another one """)
        image = Image.open("images/save_vid.png")
        st.image(image)

        st.markdown(""" 
        * When clicking on save the detection of the sign language will be displayed, a video a long with the word. Try it !!
        """)

        st.markdown("##### Option 3")
        st.markdown(""" 
        * You can record a video yourself on our website 
        * Click on the Launch Recording Webcam button to open the webcam
        *  Press on S to start the countdown 
        *  Press on Q to stop the recording 
        *  You can either record another video or save it
                        """)
        image_1 = Image.open("images/webc.png")
        st.image(image_1)
        image = Image.open("images/counter.png")
        st.image(image)


if __name__ == '__main__':
		main()	