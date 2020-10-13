import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Dense
import CreateDataset

img_data, class_name = CreateDataset.create_dataset()
target_dict = {k: v for v, k in enumerate(np.unique(class_name))}

X = img_data
X_train, X_test = train_test_split(X, train_size=0.85, random_state=7)
X_train = np.array(X_train).reshape(len(np.array(X_train)), np.prod(np.array(X_train).shape[1:]))
X_test = np.array(X_test).reshape(len(np.array(X_test)), np.prod(np.array(X_test).shape[1:]))

autoencoder = Sequential()
autoencoder.add(Dense(units=1250, activation='relu', input_dim=2500, name='encoder_layer_1'))  # encoder layers
autoencoder.add(Dense(units=312, activation='relu', input_dim=625, name='encoder_layer_2'))
autoencoder.add(Dense(units=78, activation='relu', input_dim=156, name='encoder_layer_3'))

autoencoder.add(Dense(units=156, activation='relu', name='decoder_layer_1'))  # decoder layers
autoencoder.add(Dense(units=625, activation='relu', name='decoder_layer_2'))
autoencoder.add(Dense(units=2500, activation='relu', name='decoder_layer_3'))

autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

autoencoder.summary()

autoencoder.fit(x=X_train, y=X_train, epochs=30, batch_size=64, shuffle=True,
                validation_data=(X_train, X_train), verbose=1)

encoded_input = Input(shape=(78,))
decoder_layer1 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_layer3 = autoencoder.layers[-1](decoder_layer2)

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[2].output)
encoder.save("./Models/Encoder")

decoder = Model(inputs=encoded_input, outputs=decoder_layer3)
decoder.save("./Models/Decoder")

encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(50, 50))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(13, 6))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs[i].reshape(50, 50))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
