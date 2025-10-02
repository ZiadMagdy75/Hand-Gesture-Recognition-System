import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

def prepare_folder(folder_path, max_images=None):
    all_images = []
    all_filenames = []
    
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    print(f"Found {len(subfolders)} subfolders")
    
    image_count = 0
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Processing {subfolder}: {len(image_files)} images")
        
        for img_file in image_files:
            if max_images and image_count >= max_images:
                break
                
            full_img_path = os.path.join(subfolder_path, img_file)
            try:
                img = tf.keras.preprocessing.image.load_img(full_img_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)
                
                all_images.append(processed_img[0])
                all_filenames.append(f"{subfolder}/{img_file}")
                image_count += 1
                
            except Exception as e:
                print(f"Error processing {full_img_path}: {e}")
    
    return np.array(all_images), all_filenames


train_folder_path = r'C:\Users\Access\Documents\data\data\train'
val_folder_path = r'C:\Users\Access\Documents\data\val_final'

print("Processing training folder...")
train_images, train_filenames = prepare_folder(train_folder_path) 

print("\nProcessing validation folder...")
val_images, val_filenames = prepare_folder(val_folder_path)  
print(f"\nData prepared:")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")


mobile = tf.keras.applications.MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)


x = mobile.layers[-6].output
x = GlobalAveragePooling2D()(x) 
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Added to prevent overfitting
output = Dense(units=32, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=output)

# Freezing strategy
for layer in model.layers[:-23]:
    layer.trainable = False

model.summary()
