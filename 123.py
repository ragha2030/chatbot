import datetime
import requests
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import mediapipe as mp
from math import hypot
import pyautogui
import speech_recognition as sr
import pyttsx3

# Function to get the current time
def get_current_time():
    current_time = datetime.datetime.now().strftime("%H:%M")
    return current_time

# Function to get the current date
def get_current_date():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    return current_date

# Function to get the weather
def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city}&appid={api_key}&units=metric"
    response = requests.get(complete_url)
    data = response.json()

    if data["cod"] == "404":
        return "City not found"
    elif data["cod"] == "401":
        return "Invalid API key"
    else:
        try:
            weather_info = data["main"]
            temperature = weather_info["temp"]
            humidity = weather_info["humidity"]
            weather_desc = data["weather"][0]["description"]
            return f"The weather in {city} is {weather_desc}. Temperature: {temperature}°C, Humidity: {humidity}%"
        except KeyError:
            return f"Sorry, weather information for {city} is currently unavailable."


# Emotion Detection Setup
emotion = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = keras.models.load_model("model_35_91_61.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Global variable to track if the camera should be turned off
camera_on = True

# Function to turn off the camera
def turn_off_camera():
    global camera_on
    camera_on = False
    cv2.destroyAllWindows()

# Function to turn on the camera
def turn_on_camera():
    global camera_on
    camera_on = True
    detect_emotion()

def turn_off_camera():
    global camera_on
    camera_on = False

# Volume Control Setup
def detect_hand_gesture_increase():
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    x1, y1, x2, y2 = 0, 0, 0, 0
    prev_vol_change = 0

    while camera_on:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        lmList = []

        if results.multi_hand_landmarks:
            for handlandmark in results.multi_hand_landmarks:
                for id, lm in enumerate(handlandmark.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

            if lmList != []:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]

        length = hypot(x2 - x1, y2 - y1)
        vol_change = np.interp(length, [15, 220], [-10, 10])

        # Simulate volume control using pyautogui
        if vol_change != prev_vol_change:
            pyautogui.press("volumeup" if vol_change > 0 else "volumedown")
            prev_vol_change = vol_change

        cv2.imshow('Image', img)

        key = cv2.waitKey(1)
        if key == 27:  # Esc key
            turn_off_camera()
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_emotion():
    global camera_on
    cam = cv2.VideoCapture(0)
    while camera_on:
        ret, frame = cam.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cas.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_component = gray[y:y + h, x:x + w]
                fc = cv2.resize(face_component, (48, 48))
                inp = np.reshape(fc, (1, 48, 48, 1)).astype(np.float32)
                inp = inp / 255.

                predict_prob = model.predict(inp)
                em = emotion[np.argmax(predict_prob)]
                score = np.max(predict_prob)
                cv2.putText(frame, em + "  " + str(score * 100) + '%', (x, y), font, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.imshow("image", frame)

            key = cv2.waitKey(1)
            if key == 27:  # Esc key
                # Allow emotion detection to continue if "detect emotions" command is received again
                turn_off_camera()
                while True:
                    user_input = listen_for_speech()
                    if user_input.lower() == "detect emotions":
                        break
                continue
        else:
            print('Error')
            break

    cam.release()
    cv2.destroyAllWindows()
    turn_on_camera()

def listen_for_speech():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
        return ""
    except sr.RequestError:
        print("Oops! The speech recognition service is currently unavailable.")
        return ""

# Initialize the text-to-speech engine
engine = pyttsx3.init()
def respond_to_text(text):
    if "volume" in text.lower():
        # Use hand gestures for volume control
        detect_hand_gesture_increase()  # For demonstration purposes, use increase function
        response = "Controlling volume with hand gestures."
    elif "detect emotions" in text.lower():
        # Use emotion detection
        turn_on_camera()
        response = "Detecting emotions."
    elif "hello" in text.lower():
        response = "Hello! How can I assist you?"
    elif "time" in text.lower():
        # Get the current time
        current_time = get_current_time()
        response = f"The current time is {current_time}."
    elif "date" in text.lower():
        # Get the current date
        current_date = get_current_date()
        response = f"The current date is {current_date}."
    elif "weather" in text.lower():
        # Get weather information
        api_key = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with your OpenWeatherMap API key
        city = "Chennai"  # Replace with your desired city
        weather_info = get_weather(api_key, city)
        response = weather_info
    else:
        response = "I'm sorry, I'm not sure how to respond to that."

    # Speak the response using text-to-speech
    engine.say(response)
    engine.runAndWait()

    return response



# Main Program
print("Say 'volume' to control the volume with hand gestures, 'detect emotions' to detect emotions, or 'exit' to quit.")
print("Chatbot: Hi! You can start with Hello.")     


while True:
    user_input = listen_for_speech()
        
    if user_input.lower() == "exit":
        print("Exiting...")
        turn_off_camera()
        break

    if user_input:
        print("You:", user_input)
        response = respond_to_text(user_input)
        print("Bot:", response)
