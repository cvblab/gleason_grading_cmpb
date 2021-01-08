import os
import numpy as np
from utils import *
from sklearn.utils.class_weight import compute_class_weight
import math

# GPU selection
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Seed for reproducibility
seed = 12
np.random.seed(seed)

# --- INPUTS --- #
dir_data = '../data/'
dir_results = '../data/results/'
input_shape = (224, 224, 3)
backbone = 'fsconv'
top_model = 'GMP' # vgg19, resnet, fsconv
optimizer = 'SGD'
learning_rate = 0.01
classes = ['NC', 'G3', 'G4', 'G5']
n_epochs = 100
weighting = True
batch_size = 32

# --- PREPARE OUTPUT FOLDER --- #
if not os.path.exists(dir_results + '/' ):
    os.mkdir(dir_results + '/' )

# --- DATA GENERATORS --- #

# Train generator with data augmentation
data_generator_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, featurewise_center= False,
                                                                       samplewise_center=False,
                                                                       featurewise_std_normalization=False,
                                                                       samplewise_std_normalization=False,
                                                                       zca_whitening=False,
                                                                       zca_epsilon=1e-06,
                                                                       rotation_range=90,
                                                                       width_shift_range=0.05,
                                                                       height_shift_range=0.05,
                                                                       brightness_range=[0.5, 1.5],
                                                                       shear_range=0.0,
                                                                       zoom_range=0.0,
                                                                       channel_shift_range=0.0,
                                                                       fill_mode='nearest',
                                                                       cval=0.0,
                                                                       horizontal_flip=True,
                                                                       vertical_flip=True,
                                                                       preprocessing_function=None)
train = data_generator_train.flow_from_dataframe(dataframe=pd.read_excel(dir_data+'/partition/Test/Train.xlsx'),
                                                 directory=dir_data+'/images/',
                                                 x_col='image_name',
                                                 y_col=classes,
                                                 batch_size=batch_size,
                                                 seed=42,
                                                 shuffle=True,
                                                 class_mode='raw',
                                                 target_size=input_shape[:-1])

data_generator_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test = data_generator_test.flow_from_dataframe(dataframe=pd.read_excel(dir_data+'/partition/Test/Test.xlsx'),
                                               directory=dir_data+'/images/',
                                               x_col='image_name',
                                               y_col=classes,
                                               batch_size=batch_size,
                                               seed=42,
                                               shuffle=False,
                                               class_mode='raw',
                                               target_size=input_shape[:-1])

# --- MODEL TRAINING --- #
# Create and compile model
model = get_model(backbone, top_model, input_shape=input_shape, output_shape=len(classes), optimizer_name=optimizer,
                  learning_rate=learning_rate, n_epochs=n_epochs)

# Define weights for class imbalance
if weighting:
    class_weights = compute_class_weight('balanced', [0, 1, 2, 3], np.argmax(train.labels, 1))
    class_weights = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2], 3: class_weights[3]}
else:
    class_weights = {0: 1, 1: 1, 2: 1, 3: 1}

# Train model
history = model.fit_generator(train, epochs=n_epochs, steps_per_epoch=math.ceil(train.n / batch_size),
                              validation_data=test, validation_steps=math.ceil(test.n / batch_size),
                              class_weight=class_weights)

# Reset data generators after training
test.reset()
train.reset()

# --- EVALUATION --- #

# Evaluate results in validation set
y_pred_probabilities = model.predict_generator(test, math.ceil(test.n / batch_size))  # Predictions (Prob)

# Reset data generator after prediction and set references
test.reset()
preds = np.argmax(y_pred_probabilities, axis=1)  # Predictions (Label)
refs = np.argmax(test.labels, 1)  # References

# Learning curve, confusion matrix and predictions
evaluate(refs, preds, test, history, dir_results, classes)

# Save the cnn model with json and weights
model_json = model.to_json()
with open(dir_results + 'model.json', "w") as json_file:
    json_file.write(model_json)
model.save_weights(dir_results + 'model.h5')
print('Saved model to disk')
