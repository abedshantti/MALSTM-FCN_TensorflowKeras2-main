import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Multiply, Concatenate, Activation, Masking, Reshape
from tensorflow.keras.layers import Attention, Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten, RepeatVector, Lambda

from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM, Encoder, BahdanauAttention, Decoder

DATASET_INDEX = 14

MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

TRAINABLE = True


# def generate_model():
#     ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

#     x = Masking()(ip)
#     x = LSTM(8)(x)
#     x = Dropout(0.8)(x)

#     y = Permute((2, 1))(ip)
#     y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)
#     y = squeeze_excite_block(y)

#     y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)
#     y = squeeze_excite_block(y)

#     y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)

#     y = GlobalAveragePooling1D()(y)

#     x = concatenate([x, y])

#     out = Dense(NB_CLASS, activation='softmax')(x)

#     model = Model(ip, out)
#     model.summary()

#     # add load model code here to fine-tune

#     return model

#encoder = Encoder(encoder_units, batch_size)
#enc_hidden = encoder.initialize_hidden_state()


def generate_model_2():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
    
#    encoder_units = 1024
#    batch_size = 32
    
    
    # stride = 10

    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    #ip1 = K.reshape(ip,shape=(MAX_TIMESTEPS,MAX_NB_VARIABLES))
    #x = Permute((2, 1))(ip)
    
    
    x0 = Masking()(ip)
#    x = AttentionLSTM(8)(x)
    x0 = LSTM(128)(x0)
#    x = Attention(x)

    e = Dense(1, activation='tanh')(x0)
    # Now do all the softmax business taking the above o/p
    e = Flatten()(e)
    a = Activation('softmax')(e)
    temp = RepeatVector(128)(a)
    temp = Permute([2, 1])(temp)
    # multiply weight with lstm layer o/p
    x = Multiply()([x0, temp])
    # Get the attention adjusted output state
    x = Lambda(lambda values: tf.compat.v1.keras.backend.sum(values, axis=1))(x)

    #    x = Dropout(0.8)(x)
    
#     encoder = Encoder(encoder_units, batch_size)
#     decoder = Decoder(encoder_units, batch_size)
    
#     sample_hidden = encoder.initialize_hidden_state()
    
#     sample_output, sample_hidden = encoder(x, sample_hidden)
    
#     sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
#                                       sample_hidden, sample_output)
    
#     x = Sequential()
#     x.add( Masking(mask_value=0.0, input_shape = (MAX_NB_VARIABLES, MAX_TIMESTEPS)) )
#     #x.add( AttentionLSTM(8) )
#     x.add(LSTM(8))
#     x.add( Dropout(0.8) )
    
#     y = Sequential()
#     y.add( Permute((2,1), input_shape = (MAX_NB_VARIABLES, MAX_TIMESTEPS)) )
#     y.add( Conv1D(128, 8, padding='same', kernel_initializer='he_uniform') )
#     y.add( BatchNormalization() )
#     y.add( Activation('relu') )
#     y.add( squeeze_excite_block(y.get_layer(index = -1).output, 128) )

#     y.add( Conv1D(256, 5, padding='same', kernel_initializer='he_uniform') )
#     y.add( BatchNormalization() )
#     y.add( Activation('relu') )
#     y.add( squeeze_excite_block(y.get_layer(index = -1).output, 256) )
    
#     y.add( Conv1D(128, 3, padding='same', kernel_initializer='he_uniform') )
#     y.add( BatchNormalization() )
#     y.add( Activation('relu') )

#     y.add( GlobalAveragePooling1D() )
        
        
    
    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y, 128)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y, 256)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = tf.keras.layers.Concatenate()([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model

# def generate_model_3():
#     ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

#     x = Masking()(ip)
#     x = LSTM(8)(x)
#     x = Dropout(0.8)(x)

#     y = Permute((2, 1))(ip)
#     y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)
#     #y = squeeze_excite_block(y)

#     y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)
#     #y = squeeze_excite_block(y)

#     y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)

#     y = GlobalAveragePooling1D()(y)

#     x = concatenate([x, y])

#     out = Dense(NB_CLASS, activation='softmax')(x)

#     model = Model(ip, out)
#     model.summary()

#     # add load model code here to fine-tune

#     return model


# def generate_model_4():
#     ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
#     # stride = 3
#     #
#     # x = Permute((2, 1))(ip)
#     # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
#     #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
#     # x = Permute((2, 1))(x)

#     x = Masking()(ip)
#     x = AttentionLSTM(8)(x)
#     x = Dropout(0.8)(x)

#     y = Permute((2, 1))(ip)
#     y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)
#     #y = squeeze_excite_block(y)

#     y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)
#     #y = squeeze_excite_block(y)

#     y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)

#     y = GlobalAveragePooling1D()(y)

#     x = concatenate([x, y])

#     out = Dense(NB_CLASS, activation='softmax')(x)

#     model = Model(ip, out)
#     model.summary()

#     # add load model code here to fine-tune

#     return model

def squeeze_excite_block(input, filters):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    #filters = input._keras_shape[-1] # channel_axis = -1 for TF
    
    #se = Sequential()
    
#     se.add( GlobalAveragePooling1D(input_shape = input) )
#     se.add( Reshape((1, filters)) )
#     se.add( Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False) )
#     se.add( Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False) )
#    se.add( multiply([input, se] )

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = tf.keras.layers.multiply([input, se])
    #print(se)
    
    return se


if __name__ == "__main__":
    model = generate_model_2()

    train_model(model, DATASET_INDEX, dataset_prefix='ozone', epochs=600, batch_size=128)

    evaluate_model(model, DATASET_INDEX, dataset_prefix='ozone', batch_size=128)
