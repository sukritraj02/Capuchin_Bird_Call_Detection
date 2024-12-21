def analyze_forest_recordings(model, mfcc_features):
    predictions = []
    for mfcc in mfcc_features:
        prediction = predict_bird_call(model, mfcc)
        predictions.append(prediction)
    return predictions
