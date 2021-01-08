import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix


def get_model(backbone, top_model, input_shape=(224, 224, 3), output_shape=4, optimizer_name='SGD', learning_rate=0.01,
              n_epochs=20):

    # Set backbone
    if backbone == 'vgg19':
        base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone == 'resnet':
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone == 'fsconv':
        base_model = tf.keras.Sequential()
        base_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding="same"))
        base_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        base_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
        base_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        base_model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding="same"))
        base_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Set top model
    x = base_model.layers[-1].output
    if 'GAP' in top_model:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if 'GMP' in top_model:
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax', name='predictions')(x)

    # Join base and top model
    model = tf.keras.Model(base_model.input, x)

    # Set optimizer and learning rate
    if 'SGD' in optimizer_name:
        if 'decay' in optimizer_name:
            optimizer = tf.keras.optimizers.SGD(learning_rate, decay=learning_rate / n_epochs)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate)
    elif 'Adam' in optimizer_name:
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    elif 'Adagrad' in optimizer_name:
        optimizer = tf.keras.optimizers.Adagrad(lr=learning_rate, decay=learning_rate / n_epochs)
    else:
        print('Non-valid optimizer... using SGD')
        optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Compile
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


def evaluate(refs, preds, generator, history, dir_results, labels):

    # Make plots for learning-curve
    learning_curve_plot(history, dir_results, 'lc')

    # Obtain confusion matrix (no normalized)
    ax = plot_confusion_matrix(refs, preds, np.array(labels))
    ax.figure.savefig(dir_results + '/cm')
    plt.close()


def learning_curve_plot(history, dir_out, name_out):

    plt.figure()
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.axis([0, history.epoch[-1], 0, 1])
    plt.legend(['acc', 'val_acc'], loc='upper right')
    plt.title('learning-curve')
    plt.ylabel('accuracy')
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.axis([0, history.epoch[-1], 0, max(history.history['loss'] + history.history['val_loss'])])
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(dir_out + '/' + name_out)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax