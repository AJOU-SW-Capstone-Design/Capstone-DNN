import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(100)
tf.random.set_seed(100)

file1 = './SampleData.csv'
data = pd.read_csv(file1)

train_X = data.drop(columns=['Y'])
train_Y = data['Y']

# train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2,random_state = 42)

file2 = './TestData.csv'
data2 = pd.read_csv(file2)

test_X = data2.drop(columns=['Y'])
test_Y = data2['Y']

train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y.values))
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=64)

def DNN():
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return tf_model

model = DNN()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

model_path = 'tf_dnn_model'
version = '1'
save_path = f'{model_path}/{version}'

history = model.fit(train_ds, epochs=1000, verbose=1)
print('\n')
loss,acc = model.evaluate(test_X, test_Y)
print("Loss : ", loss, "Acc : ", acc)

start = time.time()
pred_y = model.predict(test_X)
end = time.time()

print(f"{end - start:.5f} sec")

plt.scatter(test_Y, pred_y, alpha=0.4)
plt.xlabel("Actual Total Time")
plt.ylabel("Predicted Total Time")
plt.title("DNN")
plt.show()

# 주석해제
tf.keras.models.save_model(model, save_path)

tf_model = tf.keras.models.load_model('./tf_dnn_model/1')