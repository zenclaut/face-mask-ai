

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

def build_model(input_shape, num_classes, learning_rate, dropout_rate=0.2):
    
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    
    if dropout_rate > 0:
        print(f"Adding a Dropout layer with rate {dropout_rate}")
        x = Dropout(dropout_rate)(x)
   
    else:
        print("Dropout layer is disabled.")
   
    x = Dense(128, activation='relu')(x)
   
    outputs = Dense(num_classes, activation='softmax')(x)
   
    model = Model(inputs, outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
   
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_generators(train_df, val_df, img_size, batch_size, use_augmentation=False):
    if use_augmentation:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20, 
            width_shift_range=0.2, 
            height_shift_range=0.2,
            shear_range=0.2, 
            zoom_range=0.2, 
            brightness_range=[0.7, 1.3],
            horizontal_flip=True, 
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
   
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df, 
        x_col='filepath', 
        y_col='label',
        target_size=img_size, 
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle=True
    )
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df, 
        x_col='filepath', 
        y_col='label',
        target_size=img_size,
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle=False
    )
    return train_generator, val_generator


def predict_single_image(model, img_path, class_labels_map, img_size=(128, 128)):

    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    prediction = model.predict(img_preprocessed)
    
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_label = class_labels_map[predicted_class_index]
    confidence = np.max(prediction[0]) * 100

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class_label}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.show()

def plot_results(history,title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
 
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(14, 6)) 
    plt.suptitle(title)
    plt.subplot(1, 2, 1) 
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2) 
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout() 
    plt.show()



def plot_class_distribution(train_df, val_df, test_df, label_col='label'):
    train_counts = train_df[label_col].value_counts().sort_index()
    val_counts   = val_df[label_col].value_counts().sort_index()
    test_counts  = test_df[label_col].value_counts().sort_index()

    df_counts = pd.DataFrame({
        'Train': train_counts,
        'Validation': val_counts,
        'Test': test_counts
    }).fillna(0)

    colors = plt.cm.tab10(range(len(df_counts.index)))
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, cls in enumerate(df_counts.index):
        ax1.bar(cls, df_counts.loc[cls, 'Train'], color=colors[i])

    ax1.set_title('Training Set')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')

    x = range(len(df_counts.index))
    width = 0.35

    val_vals = df_counts['Validation'].values
    test_vals = df_counts['Test'].values

    ax2.bar([i - width/2 for i in x], val_vals, width=width, label='Validation')
    ax2.bar([i + width/2 for i in x], test_vals, width=width, label='Test')

    ax2.set_xticks(x)
    ax2.set_xticklabels(df_counts.index, rotation=15)
    ax2.set_title('Validation vs Test')
    ax2.set_xlabel('Class')
    ax2.legend()

    for i, v in enumerate(val_vals):
        ax2.text(i - width/2, v + 2, str(int(v)), ha='center')

    for i, v in enumerate(test_vals):
        ax2.text(i + width/2, v + 2, str(int(v)), ha='center')

    plt.tight_layout()
    plt.show()

    return df_counts

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data
def show_samples_of_data(data):
    display(data.head())

def display_samples_images(data):
    unique_labels = data['label'].unique();
    num_classes = len(unique_labels)
    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))

    for i, label in enumerate(unique_labels):
        sample_filepath = data[data['label'] == label].iloc[0]['filepath']
        
        ax = axes[i]
        
        try:
            img = Image.open(sample_filepath)
            ax.imshow(img)
            ax.set_title(f"Label: {label}")
            ax.axis('off')
        except FileNotFoundError:
            ax.set_title(f"Label: {label}\nFILE NOT FOUND")
            print(f"ERROR: Could not find image at path: {sample_filepath}")

    plt.tight_layout()
    plt.show()
    
def correct_filepaths(data):
    data['filepath']=data['filepath'].str.replace('/content','data',regex=False)
    