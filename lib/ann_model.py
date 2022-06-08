import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l2
from tensorflow.keras.utils import plot_model


class ann_model:
    def __init__(self, n_assets):
        self.num_hidden = 250
        self.n_assets = n_assets

    def keras_model(self):

        inputs = layers.Input(shape=(10, 5))
        lstm1, _, _ = layers.LSTM(32, activation="sigmoid", return_sequences=True, return_state=True)(inputs)
        # lstm1 = layers.Dense(32, activation="LeakyReLU", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inputs)
    
        inputs2 = layers.Input(shape=(10, 5))
        lstm2, _, _ = layers.LSTM(32, activation="sigmoid", return_sequences=True, return_state=True)(inputs2)
        # lstm2 = layers.Dense(32, activation="LeakyReLU", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inputs2)
    
        inputs3 = layers.Input(shape=(10, 5))
        lstm3, _, _ = layers.LSTM(32, activation="sigmoid", return_sequences=True, return_state=True)(inputs3)
        # lstm3 = layers.Dense(32, activation="LeakyReLU", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inputs3)
    
        inputs4 = layers.Input(shape=(10, 5))
        lstm4, _, _ = layers.LSTM(32, activation="sigmoid", return_sequences=True, return_state=True)(inputs4)
        # lstm4 = layers.Dense(32, activation="LeakyReLU", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inputs4)
    
        conca = layers.Concatenate()([lstm1, lstm2, lstm3, lstm4])
        her = layers.Flatten()(conca)
    
        common0 = layers.Dense(self.num_hidden, activation="LeakyReLU", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(
            her)
        common = layers.Dense(self.num_hidden, activation="LeakyReLU", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(
            common0)
    
        ac_p = layers.Dense(self.n_assets, activation="softmax")(common)
    
        critic = layers.Dense(1)(common)
    
        # model = keras.Model(inputs=[inputs, inputs2], outputs=[action_nn, ac_p, critic])
        model = keras.Model(inputs=[inputs, inputs2, inputs3, inputs4], outputs=[ac_p, critic])
    
        optimizer = keras.optimizers.Adam(learning_rate=0.009, amsgrad=True)
        # huber_loss = keras.losses.Huber()
    
        loss_function = keras.losses.MeanSquaredError()
        # mse_loss = keras.losses.Huber()

        plot_model(model, show_shapes=True, show_dtype=True, show_layer_names=True, to_file='model.png')
    
        return model, loss_function, optimizer
