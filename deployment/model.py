import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense, Activation, LeakyReLU
from tensorflow.keras.models import Model, Sequential

def get_G(input_shape):

    nin = Input(input_shape)
    n = Conv2D(64, (3, 3), (1, 1), activation='relu', padding='SAME')(nin)
    temp = n

    # B residual blocks
    for i in range(16):
        nn = Conv2D(64, (3, 3), (1, 1), padding='SAME')(n)
        nn = BatchNormalization()(nn)
        nn = Activation('relu')(nn)
        nn = Conv2D(64, (3, 3), (1, 1), padding='SAME')(nn)
        nn = BatchNormalization()(nn)
        nn = Add()([n, nn])
        n = nn

    n = Conv2D(64, (3, 3), (1, 1), padding='SAME')(n)
    n = BatchNormalization()(n)
    n = Add()([n, temp])
    # B residual blocks end

    n = Conv2D(256, (3, 3), (1, 1), padding='SAME')(n)
    n = tf.nn.depth_to_space(input=n, block_size=2)
    n = Activation('relu')(n)

    n = Conv2D(256, (3, 3), (1, 1), padding='SAME')(n)
    n = tf.nn.depth_to_space(input=n, block_size=2)
    n = Activation('relu')(n)

    nn = Conv2D(3, (1, 1), (1, 1), activation='tanh', padding='SAME')(n)
    G = Model(inputs=nin, outputs=nn, name="generator")
    return G

def get_D(input_shape):
    df_dim = 64
    lrelu = tf.keras.layers.LeakyReLU(0.2)

    nin = Input(input_shape)
    n = Conv2D(df_dim, (4, 4), (2, 2), activation=lrelu, padding='SAME')(nin)

    n = Conv2D(df_dim * 2, (4, 4), (2, 2), padding='SAME')(n)
    n = BatchNormalization()(n)
    n = Activation(lrelu)(n)
    n = Conv2D(df_dim * 4, (4, 4), (2, 2), padding='SAME')(n)
    n = BatchNormalization()(n)
    n = Activation(lrelu)(n)
    n = Conv2D(df_dim * 8, (4, 4), (2, 2), padding='SAME')(n)
    n = BatchNormalization()(n)
    n = Activation(lrelu)(n)
    n = Conv2D(df_dim * 16, (4, 4), (2, 2), padding='SAME')(n)
    n = BatchNormalization()(n)
    n = Activation(lrelu)(n)
    n = Conv2D(df_dim * 32, (4, 4), (2, 2), padding='SAME')(n)
    n = BatchNormalization()(n)
    n = Activation(lrelu)(n)
    n = Conv2D(df_dim * 16, (1, 1), (1, 1), padding='SAME')(n)
    n = BatchNormalization()(n)
    n = Activation(lrelu)(n)
    n = Conv2D(df_dim * 8, (1, 1), (1, 1), padding='SAME')(n)
    nn = BatchNormalization()(n)

    n = Conv2D(df_dim * 2, (1, 1), (1, 1), padding='SAME')(nn)
    n = BatchNormalization()(n)
    n = Activation(lrelu)(n)
    n = Conv2D(df_dim * 2, (3, 3), (1, 1), padding='SAME')(n)
    n = BatchNormalization()(n)
    n = Activation(lrelu)(n)
    n = Conv2D(df_dim * 8, (3, 3), (1, 1), padding='SAME')(n)
    n = BatchNormalization()(n)
    n = Add()([n, nn])

    n = Flatten()(n)
    no = Dense(units=1)(n)
    D = Model(inputs=nin, outputs=no, name="discriminator")
    return D
