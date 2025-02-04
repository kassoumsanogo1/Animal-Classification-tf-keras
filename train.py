import tensorflow as tf
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt


#TDownload and prepare the data
def download_and_prepare_data():
    """Télécharge et prépare les données d'entraînement et de validation."""
    #_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    #zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=_URL, extract=True)
    #base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
    #base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered_extracted')

    base_dir = os.path.join('cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    
    return train_dir, validation_dir

# Create the image generators and augment the data
def create_generators(train_dir, validation_dir, batch_size=100, img_size=150):
    """Crée les générateurs d'images pour l'entraînement et la validation."""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        #brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, validation_generator


# Build the model
def build_model(img_size=150):
    """Construit le modèle CNN pour la classification des images."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    

    #model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
    #          loss='binary_crossentropy',
    #          metrics=['accuracy'])

    
    return model



#Option 1 (standard) : batch_size = 32, epochs = 25 (bon compromis précision/stabilité).
#Option 2 (plus rapide) : batch_size = 64, epochs = 20 (plus rapide, mais besoin de GPU puissant).
#Option 3 (petite mémoire GPU) : batch_size = 16, epochs = 30 (si problème de RAM).
#Option 4 (plus précis) : batch_size = 32, epochs = 30 (plus long, mais plus précis).


# Train the model
def train_model(model, train_generator, validation_generator, epochs=25, batch_size=32):
    """Entraîne le modèle CNN sur les données d'entraînement et de validation."""
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator) // batch_size,  
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator) // batch_size,  

    )
    return history


# Plot the results
def plot_results(history):
    """Affiche les courbes de précision et de perte."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# Main function
if __name__ == "__main__":
    train_dir, validation_dir = download_and_prepare_data()
    train_gen, val_gen = create_generators(train_dir, validation_dir)
    model = build_model()
    history = train_model(model, train_gen, val_gen)
    model.save("cats_dogs_model.h5")
    plot_results(history)
