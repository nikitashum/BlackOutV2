from keras.models import load_model
import os


def encode(data):
    export_path = os.path.join(os.getcwd(), 'Models', 'Encoder')
    enc = load_model(export_path, compile=False)

    return enc.predict(data)


def decode(encoded_img):
    export_path = os.path.join(os.getcwd(), 'Models', 'Decoder')
    dec = load_model(export_path, compile=False)

    return dec.predict(encoded_img)
