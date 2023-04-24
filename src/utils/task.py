import datetime
from src.utils.speak import say


def Time():
    time = datetime.datetime.now().strftime("%H:%M")
    say(time)


def Date():
    date = datetime.date.today()
    say(date)


def Day():
    day =  datetime.datetime.now().strftime("%A")
    say(day)


def NonInputExecution(query):
    query = str(query)
    if 'time' in query:
        Time()
    elif 'date' in query:
        Date()
    elif 'day' in query:
        Day()


def InputExecution(tag, query):
    if 'wikipedia' in tag:
        name = str(query).replace("who is", "").replace("about","").replace("what is", "").replace("wikipedia", "")
        import wikipedia
        result = wikipedia.summary(name)
        say(result)
    elif 'google' in tag:
        query = str(query).replace('google','').replace('search','')
        import pywhatkit
        pywhatkit.search(query)
