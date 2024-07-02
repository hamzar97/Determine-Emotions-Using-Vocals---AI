#!/usr/bin/python3
# -*- coding: utf-8 -*-

### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
import re
import os
from collections import Counter
import altair as alt

### Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response

### Audio imports ###
from library.speech_emotion_recognition import *
### Audio imports for English ###
from library.speech_emotion_recognitionE import *
# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'

################################################################################
################################## INDEX #######################################
################################################################################

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

################################################################################
################################## RULES #######################################
################################################################################



# Audio Index
#####################   URDU   ###################
@app.route('/audio_index', methods=['POST'])
def audio_index():
    print(os.getcwd())
    # Flash message
    flash("After pressing the button above, you will have 15sec to record your audio.")
    
    return render_template('audio.html', display_button=False)

#Audio Recording
#####################   URDU   ###################
@app.route('/audio_recording', methods=("POST", "GET"))
def audio_recording():

    SER = speechEmotionRecognition()

    rec_duration = 16
    rec_sub_dir = os.path.join('vr.wav')
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    flash("The recording is over! You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.")

    return render_template('audio.html', display_button=True)

    #####################   URDU   ###################
@app.route('/play_recording', methods=("POST", "GET"))
def play_recording():
    #open a wav format audio file using default windows player
    os.system("vr.wav")
    flash("The recording is over! You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.")
    return render_template('audio.html', display_button=True)

# Audio Emotion Analysis
   #####################   URDU   ###################
@app.route('/audio_dash', methods=("POST", "GET"))
def audio_dash():

    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models/FinalModelU.hdf5')

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)
   
    # Voice Record sub dir
    rec_sub_dir = os.path.join('vr.wav')

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)
    
    # Export predicted emotions to .txt format
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')

    # Get most common emotion
    major_emotion = max(set(emotions), key=emotions.count)

    # Calculate emotion distribution
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db','audio_emotions_dist.txt'), sep=',')

    time.sleep(0.5)

    return render_template('audio_dash.html', emo=major_emotion, prob=emotion_dist)










#####################   ENGLISH   ###################
@app.route('/audio_english', methods=['POST'])
def audio_english():
    print(os.getcwd())
    # Flash message
    flash("After pressing the button above, you will have 15sec to record your audio.")
    
    return render_template('audioEnglish.html', display_button=False)


#####################   ENGLISH   ###################
@app.route('/audio_recordingE', methods=("POST", "GET"))
def audio_recordingE():

    SERE = speechEmotionRecognitionE()

    rec_duration = 16
    rec_sub_dir = os.path.join('vre.wav')
    SERE.voice_recording(rec_sub_dir, duration=rec_duration)

    flash("The recording is over! You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.")

    return render_template('audioEnglish.html', display_button=True)

#####################   ENGLISH   ###################
@app.route('/play_recordingE', methods=("POST", "GET"))
def play_recordingE():
    #open a wav format audio file using default windows player
    os.system("vre.wav")
    flash("The recording is over! You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.")
    return render_template('audioEnglish.html', display_button=True)

# Audio Emotion Analysis
#####################   ENGLISH   ###################
@app.route('/audio_dashEng', methods=("POST", "GET"))
def audio_dashEng():

    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models/audioEnglish.hdf5')

    # Instanciate new SpeechEmotionRecognition object
    SERE = speechEmotionRecognitionE(model_sub_dir)
   
    # Voice Record sub dir
    rec_sub_dir = os.path.join('vre.wav')

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SERE.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)
    
    # Export predicted emotions to .txt format
    SERE.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')


    # Get most common emotion during the interview
    major_emotion = max(set(emotions), key=emotions.count)

    # Calculate emotion distribution
    emotion_diste = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SERE._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df = pd.DataFrame(emotion_diste, index=SERE._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db','audio_emotions_distE.txt'), sep=',')

    time.sleep(0.5)

    return render_template('audio_dashEng.html', emo=major_emotion, prob=emotion_diste)


if __name__ == '__main__':
    app.run(debug=True)
