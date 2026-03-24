import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont
import os
import threading
import soundfile as sf
from io import BytesIO
import librosa
import sounddevice as sd
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier
import customtkinter
import shutil
from faster_whisper import WhisperModel
import speech_recognition as sr
camera_running = False






#os.environ["PATH"] += ";D:\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin"



if shutil.which("ffmpeg") is None:
    print("⚠️ FFmpeg not found. Please install it and add to PATH.")





print("Loading Faster-Whisper model...")
model = WhisperModel(
    "base",               # small / base / medium (base is fast & accurate)
    device="cpu",         # or "cuda" if you have NVIDIA GPU
    compute_type="int8"   # HUGE speed boost on CPU
)


# Create main window
app = ctk.CTk()
app.title("CustomTkinter Demo")
app.geometry("900x900")
app.resizable(False, False)

# Set theme
ctk.set_appearance_mode("dark")
purple = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mic_path = os.path.join(BASE_DIR, "assets", "microphone.jpg")
speaker_path = os.path.join(BASE_DIR, "assets", "speaker.jpg")

my_image = customtkinter.CTkImage(
    light_image=Image.open(mic_path),
    dark_image=Image.open(mic_path),
    size=(60, 60)
)

my_image2 = customtkinter.CTkImage(
    light_image=Image.open(speaker_path),
    dark_image=Image.open(speaker_path),
    size=(60, 60)
)


# Define a font to reuse
my_font = ctk.CTkFont(family="Arial", size=22, weight="bold")

try:
    bg_image = Image.open("circles.png")
    bg_ctk_image = ctk.CTkImage(light_image=bg_image, dark_image=bg_image, size=(500, 400))
    bg_image2 = Image.open("circles2.png")
    bg_ctk_image2 = ctk.CTkImage(light_image=bg_image, dark_image=bg_image2, size=(500, 400))
except Exception as e:
    bg_ctk_image = None
    print(f"Warning: Could not load background image: {e}")

def clear_screen(): 
    for widget in app.winfo_children(): 
        widget.destroy()

def mainMenu(): 
    clear_screen()
    
    app.configure(fg_color="#352C3D")
        
    
   
    my_font = ctk.CTkFont(family="heavitas", size=22)
        # Label
    label = ctk.CTkLabel(
        master=app,
        text="BRAVO",
        font=ctk.CTkFont(size=70, weight="bold"),
        text_color="white",
        fg_color="#493B4D",
        corner_radius=12,
        padx=50,
        pady=70,
        anchor="center",
        
        
    )
    label.place(relx=0.5, rely=0.3, anchor="center")
    
    label2 = ctk.CTkLabel(
        master=app,
        text=" ",
        text_color="white",
        fg_color="#312833",
        corner_radius=12,
        padx=80,
        pady=900,
        anchor="center",
        font=my_font
    )
    label2.place(relx=0.03, rely=0.5, anchor="center")
    # Button 1
    btn1 = ctk.CTkButton(
        master=app,
        text=" ",
        corner_radius=24,
        font=ctk.CTkFont(size=30, weight="bold"),
        image=my_image,
        fg_color="#312933",
        width=4,
        height=4,
        command=click_one,
        
    )
    btn1.place(relx=0.07, rely=0.5, anchor="center")
    
    # Button 2
    btn2 = ctk.CTkButton(
        master=app,
        text=" ",
        corner_radius=24,
        fg_color="#312933",
        font=ctk.CTkFont(size=35, weight="bold"),
        command=click_two,
        width=4,
        height=4,
        image=my_image2
    )
    btn2.place(relx=0.07, rely=0.7, anchor="center")
    

# Button functions
def click_one():
    global camera_running
    if camera_running:
        return

    camera_running = True
    threading.Thread(target=translate, daemon=True).start()
    print("Camera button clicked")
    

def click_two():
    print("world wide three")
    autoScript()


def translate():
    global camera_running
    print("Camera thread started")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera failed to open")
        camera_running = False
        return
    detector = HandDetector(maxHands=1)

    from cvzone.ClassificationModule import Classifier
    model_path = os.path.join(BASE_DIR, "model", "keras_model.h5")
    labels_path = os.path.join(BASE_DIR, "model", "labels.txt")

    classifier = Classifier(model_path, labels_path)

    offset = 20
    imgSize = 300

    labels = ["Hello","I love you","No","Okay","Please","Thank you","Yes"]

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # SAFE crop
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0 or w == 0 or h == 0:
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.putText(imgOutput, labels[index], (x, y - 30),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x + w + offset, y + h + offset),
                          (0, 255, 0), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", imgOutput)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    camera_running = False






def start_recording():
    global recording, audio_data, stream
    recording = True
    audio_data = []

    def callback(indata, frames, time, status):
        if recording:
            audio_data.append(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        callback=callback
    )
    stream.start()

    print(" Recording started")



def stop_recording():
    global recording, stream
    recording = False

    if stream:
        stream.stop()
        stream.close()
        stream = None

    if not audio_data:
        print(" No audio recorded")
        return

    print(" Transcribing...")

    audio_np = np.concatenate(audio_data, axis=0).flatten()

    segments, info = model.transcribe(
        audio_np,
        beam_size=1,
        vad_filter=True,
        temperature=0.0,
        language="en"
    )

    full_text = " ".join(seg.text.strip() for seg in segments)

    
    def update_ui():
        transcriptBox.configure(state="normal")
        transcriptBox.delete("0.0", "end")
        transcriptBox.insert("0.0", full_text)
        transcriptBox.configure(state="disabled")

    app.after(0, update_ui)






# globals I cant find them sometimes
recording = False
audio_data = []
sample_rate = 16000
stream = None
transcriptBox = None


def autoScript():
    global listening
    clear_screen()
    listening = True
    global transcriptBox
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    back_path = os.path.join(BASE_DIR, "assets", "back.jpg")
    start_path = os.path.join(BASE_DIR, "assets", "start.jpg")
    stop_path = os.path.join(BASE_DIR, "assets", "stop.jpg")



    my_back = customtkinter.CTkImage(
        light_image=Image.open(back_path), 
        dark_image=Image.open(back_path), 
        size=(60, 60))
    my_start = customtkinter.CTkImage(
        light_image=Image.open(start_path), 
        dark_image=Image.open(start_path), 
        size=(60, 60))
    my_stop = customtkinter.CTkImage(
        light_image=Image.open(stop_path), 
        dark_image=Image.open(stop_path), 
        size=(60, 60))
    

    label2 = ctk.CTkLabel(
        master=app,
        text=" ",
        text_color="white",
        fg_color="#312833",
        corner_radius=12,
        padx=80,
        pady=900,
        anchor="center",
        font=my_font
    )
    label2.place(relx=0.03, rely=0.5, anchor="center")

    label = ctk.CTkLabel(
        master=app,
        text="BRAVO",
        text_color="white",
        font=ctk.CTkFont(size=70, weight="bold"),
        fg_color="#493B4D",
        corner_radius=12,
        padx=15,
        pady=10,
        
    )
    label.place(relx=0.5, rely=0.3, anchor="center")

    record_btn = ctk.CTkButton(
        master=app,
        text=" ",
        corner_radius=24,
        fg_color="#312933",
        image=my_start,
        font=my_font,
        command= start_recording
    )
    record_btn.place(relx=0.07, rely=0.5, anchor="center")

    stoprecord_btn = ctk.CTkButton(
        master=app,
        text=" ",
        corner_radius=24,
        fg_color="#312933",
        image=my_stop,
        font=my_font,
        command= stop_recording
    )
    stoprecord_btn.place(relx=0.07, rely=0.7, anchor="center")

    

    back_btn = ctk.CTkButton(
        master=app,
        text=" ",
        corner_radius=24,
        fg_color="#312933",
        image=my_back,
        font=my_font,
        command=stop_listening
    )
    back_btn.place(relx=0.07, rely=0.3, anchor="center")

    global transcriptBox

    transcriptBox = ctk.CTkTextbox(
        master=app,
        width=420,
        height=100,
        corner_radius=12,
        font=ctk.CTkFont(size=14)
    )
    transcriptBox.place(relx=0.5, rely=0.5, anchor="center")
    transcriptBox.insert("0.0", "Press Start and speak...\n")
    transcriptBox.configure(state="disabled")

    


    #threading.Thread(target=listen_loop, daemon=True).start()

def stop_listening():
    global listening
    listening = False
    mainMenu()





mainMenu()
app.mainloop()