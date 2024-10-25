import tensorflow as tf
from helpers_proteins_128 import *

x_train, y_train, x_test, y_test = load_data()

def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(125, return_sequences=True), input_shape=input_shape))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(125, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(125, return_sequences=True)))
    model.add(tf.keras.layers.GlobalMaxPool1D())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model

num_classes = y_train.shape[1] 
input_shape = x_train.shape[1]

model = create_model(input_shape=(input_shape, 3), num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_split=0.1, epochs=50, batch_size=32)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Summary of the model
printScores(model, x_test, y_test)