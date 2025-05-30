from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Input, UpSampling2D, BatchNormalization, Activation
from keras import activations
from keras.layers.merge import concatenate
from keras.models import Model


def identity(input_image_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=input_image_shape))
    return model


def simplest(input_image_shape):
    inputs = Input(input_image_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv2 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    model = Model(input=inputs, output=conv2)
    return model


def unet_16(input_size):
    return unet(input_size, base_num_filters=16)


def unet(input_size, base_num_filters=64):
    inputs = Input(input_size)
    conv1 = Conv2D(base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(2 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(2 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(4 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(4 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(8 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(8 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(16 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(16 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(8 * base_num_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(8 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(8 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(4 * base_num_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(4 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(4 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(2 * base_num_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(2 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(2 * base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(base_num_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(base_num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1)(conv9)

    model = Model(input=inputs, output=conv10)

    return model


def srcnn(input_size, num_filters=64):
    # from: C. Dong, C. C. Loy, K. He, and X. Tang. "Learning a deep convolutional network for image super-resolution"
    # as referenced by: C. Dong, Y. Deng, C. Change Loy, and X. Tang, “Compression artifacts reduction by a deep convolutional network”
    inputs = Input(input_size)
    conv1 = Conv2D(num_filters, 3, activation='relu', padding='same')(inputs)
    conv2 = Conv2D(num_filters, 3, activation='relu', padding='same')(conv1)
    conv3 = Conv2D(1, 3, padding='same')(conv2)
    model = Model(input=inputs, output=conv3)
    return model


def ar_cnn(input_size, num_filters=64):
    # from: C. Dong, Y. Deng, C. Change Loy, and X. Tang, “Compression artifacts reduction by a deep convolutional network”
    inputs = Input(input_size)
    conv1 = Conv2D(num_filters, 3, activation='relu', padding='same')(inputs)
    conv2 = Conv2D(num_filters, 3, activation='relu', padding='same')(conv1)
    conv3 = Conv2D(num_filters, 3, activation='relu', padding='same')(conv2)
    conv4 = Conv2D(1, 3, padding='same')(conv3)
    model = Model(input=inputs, output=conv4)
    return model


def dn_cnn(input_size, depth, num_filters=64):
    # from: K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang. "Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising"
    inputs = Input(input_size)
    relu = Conv2D(num_filters, 3, activation='relu', padding='same')(inputs)
    for _ in range(1, depth):
        conv = Conv2D(num_filters, 3, padding='same')(relu)
        bn = BatchNormalization()(conv)
        relu = Activation(activations.relu)(bn)

    conv_n = Conv2D(1, 3, padding='same')(relu)
    model = Model(input=inputs, output=conv_n)
    return model


def dn_cnn_b(input_size, num_filters=64):
    return dn_cnn(input_size, depth=20, num_filters=num_filters)