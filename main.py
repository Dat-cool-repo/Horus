import RPi.GPIO as GPIO
import time
import pyaudio
import wave
import os
import requests
import cv2
import subprocess
import re
import pygame

def play_audio(filename):
    # Initialize pygame mixer for audio playback
    pygame.mixer.init()

    try:
        # Load and play the audio file
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        # Wait for the music to finish before closing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)  # Check if music is still playing

    except Exception as e:
        print(f"Error playing audio: {e}")

def scan_wifi():
    try:
        # Run iwlist to scan Wi-Fi networks
        result = subprocess.check_output(["sudo", "iwlist", "wlan0", "scan"], universal_newlines=True)
        
        # Extract SSID, BSSID, and Signal Strength using regular expressions
        networks = []
        for cell in result.split("Cell")[1:]:  # Split results by "Cell" for each network
            ssid_match = re.search(r'ESSID:"(.*?)"', cell)
            bssid_match = re.search(r'Address: ([0-9A-Fa-f:]{17})', cell)
            signal_match = re.search(r"Signal level=(-?\d+)", cell)

            ssid = ssid_match.group(1) if ssid_match else "Unknown"
            bssid = bssid_match.group(1) if bssid_match else "Unknown"
            signal = int(signal_match.group(1)) if signal_match else None
            
            networks.append({"SSID": ssid, "BSSID": bssid, "Signal Strength": signal})

        # Sort networks by signal strength (strongest first) and take the top 10
        networks = sorted(networks, key=lambda x: x["Signal Strength"], reverse=True)[:10]

        # Print detected networks
        for i, network in enumerate(networks, start=1):
            print(f"Network {i}:")
            print(f"  SSID: {network['SSID']}")
            print(f"  BSSID: {network['BSSID']}")
            print(f"  Signal Strength: {network['Signal Strength']} dBm")
            print()

        return networks

    except subprocess.CalledProcessError as e:
        print("Error running iwlist:", e)
        return []

def get_location_from_wifi(networks, api_key):
    # Prepare Wi-Fi network data
    wifi_access_points = []
    for network in networks:
        wifi_access_points.append({
            "macAddress": network["BSSID"],
            "signalStrength": network["Signal Strength"]
        })

    wifi_data = {
        "considerIp": "false",
        "wifiAccessPoints": wifi_access_points
    }

    url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={api_key}'
    try:
        response = requests.post(url, json=wifi_data)
        result = response.json()

        # Extract and print the estimated location
        if 'location' in result:
            latitude = result['location']['lat']
            longitude = result['location']['lng']
            send_coordinates_to_server(latitude, longitude)
            print(f"Estimated Location (Latitude, Longitude): {latitude}, {longitude}")
        else:
            print("Location not found.")
    except Exception as e:
        print(f"Error calling the API: {e}")

def detect_coordination():
    api_key = "AIzaSyBGLclrhHS7OZKUPKjwtH0GyW0kvDX5n_4"  # Replace with your Google API key
    wifi_networks = scan_wifi()
    if wifi_networks:
        get_location_from_wifi(wifi_networks, api_key)

def send_coordinates_to_server(latitude, longitude):
    server_url = "http://ec2-54-172-245-121.compute-1.amazonaws.com:5000/upload_coordination"  # Replace with your server URL
    data = {
        "latitude": latitude,
        "longitude": longitude
    }
    
    try:
        requests.post(server_url, json=data)
        print("sent")
    except Exception as e:
        print(f"Error sending data to the server: {e}")

# Setup GPIO
GPIO.setmode(GPIO.BCM)

on_button_pin = 2
speak_button_pin = 3
extract_text_button_pin = 14

GPIO.setup(on_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(speak_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(extract_text_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initialize variables
isOn = False
isSpeaking = False
isExtractText = False

def extractText():
    api_url = "http://ec2-54-172-245-121.compute-1.amazonaws.com:5000/extract_text"  # Replace with your server URL
    data = {
        "process": True  # This will trigger the server to process and return the audio
    }
    local_filename = "output.wav"
    try:
        # Send POST request to the server with the necessary data
        response = requests.post(api_url, json=data)

        # Check if the response is successful
        if response.status_code == 200:
            # Write the response content (audio file) to a local file
            with open(local_filename, 'wb') as f:
                f.write(response.content)
            print(f"Audio file saved as {local_filename}")
            play_audio(local_filename)
        else:
            print(f"Failed to get audio. Server responded with status code: {response.status_code}")
            print("Response:", response.text)

    except Exception as e:
        print(f"Error during the request: {e}")

SERVER_URL_IMAGE = "http://ec2-54-172-245-121.compute-1.amazonaws.com:5000/upload_frame"
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

def camera_stream():
    ret, frame = cap.read()

    # Encode the frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    
    # Create a file-like object for HTTP POST
    files = {
        "file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")
    }

    try:
        # Send the frame to the server
        print("Sent request")
        response = requests.post(SERVER_URL_IMAGE, files=files)
        local_filename = "output.wav"
        if response.status_code == 200:
            # Write the response content (audio file) to a local file
            with open(local_filename, 'wb') as f:
                f.write(response.content)
            print(f"Audio file saved as {local_filename}")
            play_audio(local_filename)
        else:
            print(f"Failed to get audio. Server responded with status code: {response.status_code}")
            print("Response:", response.text)
        print("Request response")
    except Exception as e:
        print(f"Error sending frame: {e}")

    # Show the frame locally
    #cv2.imshow("Webcam", frame)

FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate (44.1kHz)
CHUNK = 1024  # Size of each audio chunk
RECORD_SECONDS = 2  # Duration of recording
OUTPUT_FILENAME = "output.wav"  # Output file name
SERVER_URL_SOUND = "http://ec2-54-172-245-121.compute-1.amazonaws.com:5000/upload_sound"  # Replace with your server's URL

p = pyaudio.PyAudio()

def record_audio():
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    # Record for the specified duration
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save recorded audio to a WAV file
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {OUTPUT_FILENAME}")

    # Send the recorded audio file to the server
    with open(OUTPUT_FILENAME, 'rb') as f:
        files = {'file': (OUTPUT_FILENAME, f, 'audio/wav')}
        print("Begin sending")
    
        response = requests.post(SERVER_URL_SOUND, files=files, timeout=30)
        with open("output.wav", 'wb') as f:
            f.write(response.content)

        print("Done")
        play_audio("output.wav")

try:
    while True:
        # Read the GPIO pins
        isSpeaking = GPIO.input(speak_button_pin) == 0
        if GPIO.input(on_button_pin) == 0:
            isOn = not isOn
            time.sleep(0.5)

        if GPIO.input(extract_text_button_pin) == 0:
            isExtractText = True

        # Print pin states
        print(f"On Button: {'Pressed' if isOn == True else 'Released'}")
        print(f"Speak Button: {'Pressed' if GPIO.input(speak_button_pin) == 0 else 'Released'}")
        
        # Perform actions if isSpeaking is True and isOn is True
        if isOn and not isSpeaking and not isExtractText:
            camera_stream()
            time.sleep(3)
        if isSpeaking and isOn:
            #detect_coordination()
            record_audio()  # Record the audio
        if isOn and isExtractText:
            extractText()
            isExtractText = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.5)

except KeyboardInterrupt:
    GPIO.cleanup()
    print("GPIO cleaned up.")
