from tabnanny import verbose
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
import speech_recognition as sr

# def predict_json(pred):
#     emotions = ["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"]
#     result = {}
#     for k in range(len(emotions)):
#         result[emotions[k]] = pred[k]
#     return result


def predict_emotions_text(sentence, verbose = False):
    # We do the predictions this way to avoid problems with the memory restrictions
    models = ['roberta-large-nli-stsb-mean-tokens', 'distilbert-base-nli-mean-tokens', 'bert-large-nli-stsb-mean-tokens']
    embeddings = []
    classifiers = []
    for model_name in models :
        model = SentenceTransformer(model_name)
        sentence_embeddings = model.encode(sentence)
        # We use the same models trained in the experimental phase
        clf = pickle.load(open('models/Audio/{0}.clf'.format(model_name),'rb'))
        embeddings.append(clf.predict(sentence_embeddings))

    weak_predictions = [np.concatenate((embeddings[0][i], np.concatenate((embeddings[1][i], embeddings[2][i]), axis=0)), axis=0) for i in range(len(embeddings[0]))]
    clf = pickle.load(open('models/Audio/{0}.clf'.format('Meta_learner'),'rb'))
    pred = clf.predict(weak_predictions)[0]

    emotions = ["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"]
    result = {}
    for k in range(len(emotions)):
        result[emotions[k]] = pred[k]
    if verbose:
        print(result)

    return pred


def extract_audio(filepath):
    r = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
        return s
    except Exception as e:
        print("Exception: " + str(e))
        return 'ERROR'

def predict_emotions_audio(filepath, vector_multimodal = False):

    phrase = extract_audio(filepath)
    if phrase == 'ERROR':
        return 'ERROR'

    return  predict_emotions_text([phrase], verbose = verbose)

    
