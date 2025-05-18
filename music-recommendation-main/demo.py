from tensorflow.keras.models import load_model

try:
    model = load_model('emotion_model.h5', compile=False)
    model.save('emotion_model.keras')
    model.save('clean_emotion_model.h5')
    test_model = load_model('emotion_model.keras')
    test_model.summary()
except Exception as e:
    print("Error occurred while processing the model:")
    print(e)
