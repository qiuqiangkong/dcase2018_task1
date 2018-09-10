from keras.models import Model
from keras.layers import (Input, Reshape, Dense, Conv2D, MaxPooling2D, 
                          BatchNormalization, Activation, GlobalMaxPooling2D)


def BaselineCnn(seq_len, mel_bins, classes_num):
    
    data_format = 'channels_first'
    
    input_layer = Input(shape=(seq_len, mel_bins))
    x = Reshape((1, seq_len, mel_bins))(input_layer)
    
    x = Conv2D(64, kernel_size=(5, 5), activation='linear', padding='same', data_format=data_format)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
    
    x = Conv2D(128, kernel_size=(5, 5), activation='linear', padding='same', data_format=data_format)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
    
    x = Conv2D(256, kernel_size=(5, 5), activation='linear', padding='same', data_format=data_format)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
    
    x = Conv2D(512, kernel_size=(5, 5), activation='linear', padding='same', data_format=data_format)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
    
    x = GlobalMaxPooling2D(data_format=data_format)(x)
    output_layer = Dense(classes_num, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model
    

def VggishConvBlock(input, filters, data_format):
    
    if data_format == 'channels_first':
        bn_axis = 1
    
    else:
        raise Exception('Only support channels_first now!')
    
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='linear', padding='same', data_format=data_format)(input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='linear', padding='same', data_format=data_format)(input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    output = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
    
    return output


def Vggish(seq_len, mel_bins, classes_num):
    
    data_format = 'channels_first'
    
    input_layer = Input(shape=(seq_len, mel_bins))
    x = Reshape((1, seq_len, mel_bins))(input_layer)
    
    x = VggishConvBlock(input=x, filters=64, data_format=data_format)
    x = VggishConvBlock(input=x, filters=128, data_format=data_format)
    x = VggishConvBlock(input=x, filters=256, data_format=data_format)
    x = VggishConvBlock(input=x, filters=512, data_format=data_format)
    
    x = GlobalMaxPooling2D(data_format=data_format)(x)
    x = Dense(classes_num, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=x)
    
    return model