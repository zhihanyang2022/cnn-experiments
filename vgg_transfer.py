# vgg.py
# transfer learning for cats_and_dogs_dataset
# things to change to make things work

# <codecell>
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# <codecell>
# 'include_top'
# - whether to include the densely connected classifier on top of the network
# - the default classifier corresponds to the 1000 classes form ImageNet
# - since we want to do different classification, we want to omit the classifier
# 'input_shape'
# - the shape of the image tensors that you'll feed to the network
# - if you don't pass it, the network will be able to process inputs of any shape

# <codecell>
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# set up tags for data lookup
base_dir = '/Users/yangzhihan/datasets/cats_and_dogs_dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255) # normalize image
batch_size = 10  # change to 20

def extract_features(directory, sample_count):

    features = np.zeros(shape=(sample_count, 4, 4, 512))
    # call 'conv_base.summary()' and you will see that the shape of the output of the final MaxPooling2D is (None, 4, 4, 512)

    labels = np.zeros(shape=(sample_count))
    # a label is an n-dim vector with n=the number of categories to classify

    generator = datagen.flow_from_directory(directory, target_size=(150, 150), batch_size=batch_size, class_mode='binary')
    # - directory: each folder (train, validation and test) should contain each subdirectory per class
    # - target_size: the dims to which all images found will be resized
    # - class_mode: "binary" will be 1D binary labels

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        print(features_batch.shape)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 10) # change to 2000
validation_features, validation_labels = extract_features(validation_dir, 10) # change to 1000
test_features, test_labels = extract_features(test_dir, 10) # change to 1000

# <codecell>
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Flatten(input_shape=(4, 4, 512)))
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_features, train_labels, epochs=30, batch_size=20, validation_data=(validation_features, validation_labels))

model.save("vgg_cats_and_dogs.h5")























# save history at some place

# end
