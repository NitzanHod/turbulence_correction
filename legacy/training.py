import tensorflow as tf
from legacy.model import U_Net
from legacy.data_loader import create_dataset
import config
import datetime
import time

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

dataset = create_dataset(data_main_path_directory=config.data_main_path_directory)
model = U_Net()
model.compile(optimizer='adam',
              loss=tf.keras.losses.MSE)

tic = time.time()
model_history = model.fit(dataset,
                          epochs=5,
                          callbacks=[tensorboard_callback])

toc = time.time() - tic
print('Elapsed Time: ', round(toc, 2), 'secs')
