import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output
from tensorflow.keras.layers import Dense, Dropout, Batchnormalization
from keras.models import Sequential
import tensorflow as tf


def split_data(simulation, N):
    inputs = simulation['inputs']
    outputs = simulation['outputs']
    split_idx = int(0.8 * N)
    X_train, X_val = inputs[:split_idx], inputs[split_idx:]
    y_train, y_val = outputs[:split_idx], outputs[split_idx:]

    return X_train, y_train, X_val, y_val


# Función para graficar el historial
def plot_training(history, model_name):
        clear_output(wait=True)
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=history.history['loss'], label='Training Loss', color='#1f77b4', linewidth=2.5)
        sns.lineplot(data=history.history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2.5, linestyle='--')
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss (MSE)', fontsize=16)
        plt.title(f'Training and Validation Loss for {model_name}', fontsize=18)
        plt.legend(loc='upper right', fontsize=14)
        plt.tight_layout()
        plt.show()

# Función para entrenar el modelo
def train_model(simulation, dist=('normal', 'normal'), corr=False):
    X_train, y_train, X_val, y_val = split_data(simulation, 10000)
    model = Sequential()
    
    if dist == ('normal', 'normal') and not corr:
        model.add(Dense(16, activation='relu', input_shape=(4,)))
        model.add(Dropout(0.1))
        model.add(Dense(20))
        model.compile(optimizer='adam', loss='mse')
        epochs = 20
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        return model, history
        

    elif dist == ('normal', 'normal') and corr:
        model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001),
                        input_shape=(5,)))
        model.add(Batchnormalization())

        # Arquitectura más ancha en lugar de más profunda (mejor para relaciones funcionales)
        model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(Batchnormalization())
        model.add(Dropout(0.1))  

        # Capa de salida
        model.add(Dense(20))

        # Programación de tasa de aprendizaje personalizada
        initial_learning_rate = 0.01  # Tasa inicial más alta
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True)

        # Compilación con optimizador adaptativo
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        # Callbacks para mejor entrenamiento
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Mayor paciencia ya que la convergencia es gradual después de las primeras épocas
            restore_best_weights=True
        )

        # Entrenamiento
        history = model.fit(
            X_train, y_train,
            epochs=50,  # Más épocas para permitir convergencia completa
            batch_size=128,  # Batch más grande para estabilidad
            validation_data=(X_val, y_val),
            callbacks=[early_stopping])
        return model, history
        

        
    elif dist == ('Gamma', 'Gamma') and not corr:
        model.add(Dense(16, activation='relu', input_shape=(4,)))
        model.add(Dropout(0.4))
        model.add(Dense(20))
        epochs = 50
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        return model, history
       
    else:
        model.add(Dense(16, activation='relu', input_shape=(4,)))
        model.add(Dropout(0.2))
        model.add(Dense(20))
        epochs = 30
    
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        return model, history


def mse_calculation(y,y_h):
    mse = 0
    n = len(y)
    for i in range(n):
        mse += (y[i]- y_h[i])**2

    return mse/n