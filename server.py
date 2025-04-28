from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import threading
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import torch
from scipy.signal import find_peaks, butter, filtfilt
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from ultralytics import YOLO
import simpleaudio as sa  # For playing siren sound

app = Flask(__name__)

# Load models
model = YOLO('yolov8n.pt')
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Constants
SIREN_FREQ_RANGE = (600, 1400)
TRAFFIC_THRESHOLD = 15
EMERGENCY_OVERRIDE_DURATION = 30
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 5

# Traffic state variables
traffic_signal = "RED"
emergency_override = False
override_end_time = 0
emergency_status = "No Emergency"
video_path = None
cap = None

# Sound to play when emergency detected (must have siren.wav in your project folder)
def play_siren():
    try:
        wave_obj = sa.WaveObject.from_wave_file("audio.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print("Error playing siren sound:", e)

# Function for audio detection
def detect_audio():
    global emergency_override, override_end_time, emergency_status
    while True:
        print("Listening for distress signals and sirens...")
        recording = sd.rec(int(AUDIO_DURATION * AUDIO_SAMPLE_RATE), samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        wav.write("audio.wav", AUDIO_SAMPLE_RATE, recording)
        
        _, speech = wav.read("audio.wav")
        speech = np.array(speech, dtype=np.float32)
        if speech.ndim == 1 and speech.shape[0] > 10:
            speech = speech / np.max(np.abs(speech)) if np.max(np.abs(speech)) != 0 else speech
        else:
            continue

        input_values = audio_processor(speech, return_tensors="pt", sampling_rate=AUDIO_SAMPLE_RATE).input_values
        logits = audio_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = audio_processor.batch_decode(predicted_ids)[0]
        print("Detected Speech:", transcription)
        
        distress_detected = "help" in transcription.lower() or "emergency" in transcription.lower()
        
        fft_spectrum = np.fft.fft(speech)
        frequencies = np.fft.fftfreq(len(fft_spectrum), d=1/AUDIO_SAMPLE_RATE)
        magnitudes = np.abs(fft_spectrum)
        
        b, a = butter(4, [SIREN_FREQ_RANGE[0]/(AUDIO_SAMPLE_RATE/2), SIREN_FREQ_RANGE[1]/(AUDIO_SAMPLE_RATE/2)], btype='band')
        filtered_speech = filtfilt(b, a, speech)
        fft_filtered = np.fft.fft(filtered_speech)
        magnitudes_filtered = np.abs(fft_filtered)
        
        peaks, _ = find_peaks(magnitudes_filtered, height=np.max(magnitudes_filtered) * 0.6)
        peak_frequencies = frequencies[peaks]
        
        siren_detected = any(SIREN_FREQ_RANGE[0] <= abs(freq) <= SIREN_FREQ_RANGE[1] for freq in peak_frequencies)
        
        if distress_detected or siren_detected:
            print("Emergency detected! Turning signal GREEN for", EMERGENCY_OVERRIDE_DURATION, "seconds")
            emergency_override = True
            override_end_time = time.time() + EMERGENCY_OVERRIDE_DURATION
            emergency_status = "Emergency Detected!"
            threading.Thread(target=play_siren, daemon=True).start()
        else:
            if not emergency_override:
                emergency_status = "No Emergency"

# Start audio detection thread
audio_thread = threading.Thread(target=detect_audio, daemon=True)
audio_thread.start()

# Video stream generator
def generate_frames():
    global cap, traffic_signal, emergency_override, override_end_time, emergency_status
    while True:
        if cap is None:
            continue
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        detections = results[0].boxes.data
        
        vehicle_count = 0
        pedestrian_count = 0

        for box in detections:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if int(cls) in [2, 3, 5, 7]:  # Car, motorcycle, bus, truck
                vehicle_count += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            elif int(cls) == 0:  # Pedestrian
                pedestrian_count += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        if emergency_override and time.time() < override_end_time:
            traffic_signal = "GREEN"
        elif vehicle_count > TRAFFIC_THRESHOLD or pedestrian_count > 5:
            traffic_signal = "GREEN"
            emergency_status = "No Emergency"
        else:
            traffic_signal = "RED"
            emergency_status = "No Emergency" if not emergency_override else emergency_status

        # Display info on frame
        cv2.putText(frame, f"Traffic Signal: {traffic_signal}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if traffic_signal == "RED" else (0, 255, 0), 2)
        cv2.putText(frame, f"Emergency Status: {emergency_status}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255) if "Emergency" in emergency_status else (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count} | Pedestrians: {pedestrian_count}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.circle(frame, (50, 200), 20, (0, 0, 255) if traffic_signal == "RED" else (0, 255, 0), -1)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    global cap, video_path
    if request.method == 'POST':
        file = request.files['file']
        if file:
            video_path = "./uploaded_video.mp4"
            file.save(video_path)
            cap = cv2.VideoCapture(video_path)
            return redirect(url_for('video_feed'))
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
