from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
import speech_recognition as sr
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

model_dict = {'roberta-large-nli-stsb-mean-tokens' : SentenceTransformer('roberta-large-nli-stsb-mean-tokens'),
              'distilbert-base-nli-mean-tokens' : SentenceTransformer('distilbert-base-nli-mean-tokens'),
              'bert-large-nli-stsb-mean-tokens' : SentenceTransformer('bert-large-nli-stsb-mean-tokens')}

classifier_dict = {'roberta-large-nli-stsb-mean-tokens' : pickle.load(open('{0}/assets/{1}.clf'.format(dir_path, 'roberta-large-nli-stsb-mean-tokens'),'rb')),
                   'distilbert-base-nli-mean-tokens' : pickle.load(open('{0}/assets/{1}.clf'.format(dir_path, 'distilbert-base-nli-mean-tokens'),'rb')),
                   'bert-large-nli-stsb-mean-tokens' : pickle.load(open('{0}/assets/{1}.clf'.format(dir_path, 'bert-large-nli-stsb-mean-tokens'),'rb')),
                   'meta-learner' : pickle.load(open('{0}/assets/{1}.clf'.format(dir_path, 'Meta_learner'),'rb'))}

def predict_emotions_text(sentence, verbose = False):
    models = ['roberta-large-nli-stsb-mean-tokens', 'distilbert-base-nli-mean-tokens', 'bert-large-nli-stsb-mean-tokens']
    embeddings = []
    classifiers = []
    sentences = [sentence]
    for model_name in models :
        sentence_embeddings = model_dict[model_name].encode(sentences)
        # We use the same models trained in the experimental phase
        embeddings.append(classifier_dict[model_name].predict(sentence_embeddings))

    weak_predictions = [np.concatenate((embeddings[0][i], np.concatenate((embeddings[1][i], embeddings[2][i]), axis=0)), axis=0) for i in range(len(embeddings[0]))]
    pred = classifier_dict['meta-learner'].predict(weak_predictions)[0]

    emotions = ["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"]
    result = {}
    for k in range(len(emotions)):
        result[emotions[k]] = pred[k]
    if verbose:
        print(result)

    return result

def predict_emotions_text_multiple(sentence_list, verbose = False):
    models = ['roberta-large-nli-stsb-mean-tokens', 'distilbert-base-nli-mean-tokens', 'bert-large-nli-stsb-mean-tokens']
    embeddings = []
    classifiers = []
    for model_name in models :
        sentence_embeddings = model_dict[model_name].encode(sentence_list)
        # We use the same models trained in the experimental phase
        embeddings.append(classifier_dict[model_name].predict(sentence_embeddings))

    weak_predictions = [np.concatenate((embeddings[0][i], np.concatenate((embeddings[1][i], embeddings[2][i]), axis=0)), axis=0) for i in range(len(embeddings[0]))]
    pred_list = classifier_dict['meta-learner'].predict(weak_predictions)

    emotions = ["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"]
    results = []
    for pred in pred_list:
        result = {}
        for k in range(len(emotions)):
            result[emotions[k]] = pred[k]
        if verbose:
            print(result)
        results.append(result)

    return results

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

def predict_emotions_audio(filepath, verbose = False):

    phrase = extract_audio(filepath)
    if phrase == 'ERROR':
        return 'ERROR'

    return  predict_emotions_text(phrase, verbose = verbose)

    
