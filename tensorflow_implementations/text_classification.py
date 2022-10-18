import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# load data
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train')
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

raw_train_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=32, validation_split=0.2, subset='training', seed=42)
raw_val_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=32, validation_split=0.2, subset='validation', seed=42)
raw_test_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/test', batch_size=32)
train_text = raw_train_ds.map(lambda x, y: x)

# TextVectorization layer is useful for pre-processing
vectorize_layer = layers.TextVectorization(standardize=custom_standardization, max_tokens=10000, output_mode='int', output_sequence_length=250)
vectorize_layer.adapt(train_text)

# have a look at the pre-processed data
text_batch, label_batch = next(iter(raw_train_ds))
print("Review", text_batch[0])
print("Label", raw_train_ds.class_names[label_batch[0]])
print("Vectorized review", vectorize_text(text_batch[0], label_batch[0]))
print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# pre-process all three data sets
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# protect against I/O bottleneck
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# create model
model = tf.keras.Sequential([
  layers.Embedding(10000 + 1, 16),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()
model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# train
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# evaluate
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

