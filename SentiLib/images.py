import images_dependencies

def predict_emotion_image(filepath, verbose = False):
    args = images_dependencies.Arguments(inputfile = filepath)
    processor = images_dependencies.Processor(args)
    pred = processor.start()
    emotions = ['joy','trust','fear','surprise','sadness','disgust','anger','anticipation']
    pred_dict = {}
    for k in range(len(emotions)):
        pred_dict[emotions[k]] = int(pred[0][k])
    if verbose:
        print(pred_dict)
    return pred_dict