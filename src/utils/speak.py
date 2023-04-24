import pyttsx3


def say(sentence):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 120)
    print(" ")
    print(f"SAM : {sentence}")
    engine.say(text=sentence)
    engine.runAndWait()
    print(" ")

