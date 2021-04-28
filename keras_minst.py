# jupyter 환경에서 진행된 코드
from IPython.core.display import display, HTML
display(HTML("<style>.container {width:100% !important;}</style>"))

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

# 픽셀 값을 0~1 사이로 정규화
x_train, x_test = x_train / 255., x_test / 255.

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = models.Sequential()
model.add(layers.Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(10, (5, 5), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(15, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary() # 모델 구조 확인

# 전체 데이터 중 일부만 사용
train_size = x_train.shape[0]
batch_size = 1000
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
y_batch = y_train[batch_mask]

# validation data split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, shuffle=True)

# train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x_batch, y_batch, epochs=100, verbose=1, validation_data=(x_valid, y_valid))

# test
test_size = x_test.shape[0]
test_batch_size = 100
test_batch_mask = np.random.choice(test_size, test_batch_size)

x_test_batch = x_test[test_batch_mask]
y_test_batch = y_test[test_batch_mask]

score = model.evaluate(x_test, y_test)

print("Test loss:", score[0])
print("Test accuracy:", score[1])
