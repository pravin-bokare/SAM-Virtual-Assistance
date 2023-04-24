import speech_recognition as sr
#import pyaudio


def listen():
    l = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        l.pause_threshold = 1
        audio = l.listen(source,0,4)

    try:
        print("Recognizing...")
        query = l.recognize_google(audio, language="en-in")
        print(f"Pravin : {query}")
    except Exception as e:
        return " "
    query = str(query)
    return query.lower()
