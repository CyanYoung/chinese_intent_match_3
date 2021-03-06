from keras.layers import Dense, SeparableConv1D, LSTM, Lambda
from keras.layers import GlobalMaxPooling1D, Bidirectional, Subtract, Concatenate

import keras.backend as K


def dnn(embed_input1, embed_input2, embed_input3):
    mean = Lambda(lambda a: K.mean(a, axis=1))
    da1 = Dense(200, activation='relu', name='encode1')
    da2 = Dense(200, activation='relu', name='encode2')
    norm = Lambda(lambda a: K.sum(K.square(a), axis=-1, keepdims=True))
    x = mean(embed_input1)
    x = da1(x)
    x = da2(x)
    y = mean(embed_input2)
    y = da1(y)
    y = da2(y)
    z = mean(embed_input3)
    z = da1(z)
    z = da2(z)
    pos = norm(Subtract()([x, y]))
    neg = norm(Subtract()([x, z]))
    return Subtract()([pos, neg])


def dnn_encode(embed_input):
    mean = Lambda(lambda a: K.mean(a, axis=1))
    da1 = Dense(200, activation='relu', name='encode1')
    da2 = Dense(200, activation='relu', name='encode2')
    x = mean(embed_input)
    x = da1(x)
    return da2(x)


def cnn(embed_input1, embed_input2, embed_input3):
    ca1 = SeparableConv1D(filters=64, kernel_size=1, padding='same', activation='relu', name='conv1')
    ca2 = SeparableConv1D(filters=64, kernel_size=2, padding='same', activation='relu', name='conv2')
    ca3 = SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv3')
    mp = GlobalMaxPooling1D()
    da = Dense(200, activation='relu', name='encode')
    norm = Lambda(lambda a: K.sum(K.square(a), axis=-1, keepdims=True))
    x1 = ca1(embed_input1)
    x1 = mp(x1)
    x2 = ca2(embed_input1)
    x2 = mp(x2)
    x3 = ca3(embed_input1)
    x3 = mp(x3)
    x = Concatenate()([x1, x2, x3])
    x = da(x)
    y1 = ca1(embed_input2)
    y1 = mp(y1)
    y2 = ca2(embed_input2)
    y2 = mp(y2)
    y3 = ca3(embed_input2)
    y3 = mp(y3)
    y = Concatenate()([y1, y2, y3])
    y = da(y)
    z1 = ca1(embed_input3)
    z1 = mp(z1)
    z2 = ca2(embed_input3)
    z2 = mp(z2)
    z3 = ca3(embed_input3)
    z3 = mp(z3)
    z = Concatenate()([z1, z2, z3])
    z = da(z)
    pos = norm(Subtract()([x, y]))
    neg = norm(Subtract()([x, z]))
    return Subtract()([pos, neg])


def cnn_encode(embed_input):
    ca1 = SeparableConv1D(filters=64, kernel_size=1, padding='same', activation='relu', name='conv1')
    ca2 = SeparableConv1D(filters=64, kernel_size=2, padding='same', activation='relu', name='conv2')
    ca3 = SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv3')
    mp = GlobalMaxPooling1D()
    da = Dense(200, activation='relu', name='encode')
    x1 = ca1(embed_input)
    x1 = mp(x1)
    x2 = ca2(embed_input)
    x2 = mp(x2)
    x3 = ca3(embed_input)
    x3 = mp(x3)
    x = Concatenate()([x1, x2, x3])
    return da(x)


def rnn(embed_input1, embed_input2, embed_input3):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, name='encode1')
    mp = GlobalMaxPooling1D()
    da = Dense(200, activation='relu', name='encode2')
    norm = Lambda(lambda a: K.sum(K.square(a), axis=-1, keepdims=True))
    x = ba(embed_input1)
    x = mp(x)
    x = da(x)
    y = ba(embed_input2)
    y = mp(y)
    y = da(y)
    z = ba(embed_input3)
    z = mp(z)
    z = da(z)
    pos = norm(Subtract()([x, y]))
    neg = norm(Subtract()([x, z]))
    return Subtract()([pos, neg])


def rnn_encode(embed_input):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, name='encode1')
    mp = GlobalMaxPooling1D()
    da = Dense(200, activation='relu', name='encode2')
    x = ba(embed_input)
    x = mp(x)
    return da(x)
