import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import speech_recognition as sr
import pyttsx3
import json
import random

model = load_model('chatbot_model.h5')
intents = json.loads(open('./myIntents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res)]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    
    if float(ints[0]['probability']) > 0.65:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result
    else:
        return False

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def send(msg):
    if msg != '':    
        return chatbot_response(msg)


#constructing our chatbot

bot_name = "Voicisstant"


r = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
newVoiceRate = 145
engine.setProperty('rate',newVoiceRate)
engine.setProperty('voice', voices[1].id)

def talk(text):
    engine.say(text)
    engine.runAndWait()

resp = send('hello')
talk(resp)
print(f"{bot_name}: {resp}")
print("Say 'quit' to exit.")
print('Listening...')

while True:
    # obtain audio from the microphone

    with sr.Microphone() as source:
        
        r.adjust_for_ambient_noise(source, duration = 1)
        audio = r.listen(source)
        
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        sentence = r.recognize_google(audio)
        print(f"{bot_name} thinks you said '{sentence}'")
        response = send(sentence)
        if response == False:
            for i in intents['intents']:
                if i['tag']== 'defaultfallback':
                    response = random.choice(i['responses'])
                    
        print(f"{bot_name}: {response}")
        talk(response)
        if sentence == 'quit':
            quit()
    except sr.UnknownValueError:
        print()
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

base.mainloop()
