import numpy as np
import pandas as pd
import torch
import pickle
from text import predict_emotions_text
from images import predict_emotion_image
from multimodal_dependencies import FusionTransformer, EmbraceNet, Wrapper

def activate(t):
    return (t >= 0.1).int().tolist()[0]

def predict_multimodal(sentence, filepath, verbose = False):
    text_prediction_json = predict_emotions_text(sentence, verbose=False)
    image_prediction_json = predict_emotion_image(filepath, verbose=False)

    with open('SentiLib/assets/fusion_model.pt', 'rb') as dic:
        multimodal_model = pickle.load(dic)
    
    image_vect = torch.from_numpy(np.array([[float(image_prediction_json["joy"]),float(image_prediction_json["sadness"]),float(image_prediction_json["anger"]),float(image_prediction_json["fear"]),float(image_prediction_json["surprise"]),float(image_prediction_json["anticipation"]),float(image_prediction_json["disgust"]),float(image_prediction_json["trust"])]])).float()

    text_vect = torch.from_numpy(np.array([[float(text_prediction_json["joy"]),float(text_prediction_json["sadness"]),float(text_prediction_json["anger"]),float(text_prediction_json["fear"]),float(text_prediction_json["surprise"]),float(text_prediction_json["anticipation"]),float(text_prediction_json["disgust"]),float(text_prediction_json["trust"])]])).float()
    
    prediction = activate(multimodal_model(image_vect,torch.from_numpy(np.array([[0., 0., 0., 0., 0., 0., 0., 0.]])).float(),text_vect,torch.from_numpy(np.array([[1., 0., 1.]])).float()))

    emotions = ["joy","sadness","anger","fear","surprise","anticipation","disgust","trust"]
    output = {}

    for k in range(len(emotions)):
        output[emotions[k]] = prediction[k]
    if verbose:
        print(output)
    return output

