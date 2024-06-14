# Blink_Drive_Arrive
Driver Drowsiness Detection

# Video Demonstration (Turn on the video sound to hear the alert)

https://github.com/stefanus-ai-tech/Blink_Drive_Arrive/assets/148773999/5fc62a1f-affe-4759-951f-fbcd12dcba2c

The dataset is from https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset

This repository contains code to detect driver drowsiness using a webcam feed. The model identifies whether the driver's eyes are open or closed and triggers an alert sound if closed eyes are detected for a consecutive number of frames.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Real-Time Detection](#real-time-detection)
- [Limitations](#limitations)
- [Acknowledgments](#acknowledgments)

## Prerequisites

- Python 3.x
- OpenCV
- TensorFlow
- NumPy
- Pygame
- Matplotlib
- A compatible webcam

Ensure your camera angle is correct to capture clear side view images of the driver's eyes for accurate detection.

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Create a `requirements.txt` file with the following contents:
   ```
   opencv-python
   tensorflow
   numpy
   pygame
   matplotlib
   ```

## Dataset Preparation

1. Organize your dataset as follows:
   ```
   dataset/
   ├── Closed_Eyes/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── Open_Eyes/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

2. Update the paths in the script to point to your dataset:
   ```python
   closed_eyes_path = '/path/to/dataset/Closed_Eyes'
   open_eyes_path = '/path/to/dataset/Open_Eyes'
   ```

## Model Training

1. Load and preprocess the images:
   ```python
   closed_eyes_images, closed_eyes_labels = load_images_from_folder(closed_eyes_path)
   open_eyes_images, open_eyes_labels = load_images_from_folder(open_eyes_path)

   images = np.concatenate((closed_eyes_images, open_eyes_images), axis=0)
   labels = np.concatenate((closed_eyes_labels, open_eyes_labels), axis=0)

   images = images / 255.0
   ```

2. Display some sample images:
   ```python
   display_images(images, labels, num_images=10)
   ```

3. Train the model using K-Fold Cross Validation:
   ```python
   for train_index, val_index in kf.split(images):
       X_train, X_val = images[train_index], images[val_index]
       y_train, y_val = labels[train_index], labels[val_index]

       datagen.fit(X_train)

       model = Sequential([
           Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
           MaxPooling2D((2, 2)),
           Conv2D(64, (3, 3), activation='relu'),
           MaxPooling2D((2, 2)),
           Flatten(),
           Dense(128, activation='relu'),
           Dropout(0.5),
           Dense(1, activation='sigmoid')
       ])

       model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

       model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                 epochs=10, 
                 validation_data=(X_val, y_val),
                 callbacks=[early_stopping])
   ```

4. Save the trained model:
   ```python
   model.save('/path/to/save/drowsiness_model.keras')
   ```

## Real-Time Detection

1. Initialize Pygame mixer and load the alert sound:
   ```python
   pygame.mixer.init()
   alert_sound = pygame.mixer.Sound('/path/to/siren-alert-96052.mp3')
   ```

2. Load the trained model:
   ```python
   model = tf.keras.models.load_model('/path/to/save/drowsiness_model.keras')
   ```

3. Start the webcam and perform real-time detection:
   ```python
   cap = cv2.VideoCapture(0)
   
   while True:
       ret, frame = cap.read()
       if not ret:
           break

       resized_frame = cv2.resize(frame, (64, 64))
       normalized_frame = resized_frame / 255.0
       reshaped_frame = np.reshape(normalized_frame, (1, 64, 64, 3))

       prediction = model.predict(reshaped_frame)
       if prediction < 0.5:
           label = 'Closed Eyes'
           closed_eyes_counter += 1
       else:
           label = 'Open Eyes'
           closed_eyes_counter = 0
           stop_sound()

       if closed_eyes_counter >= closed_eyes_threshold:
           threading.Thread(target=play_sound).start()
           closed_eyes_counter = 0

       cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       cv2.imshow('Driver Drowsiness Detection', frame)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cap.release()
   cv2.destroyAllWindows()
   ```

## Limitations

- The accuracy of the detection depends on the angle and clarity of the camera feed.
- Ensure proper lighting conditions to improve detection accuracy.
- The model may not perform well in diverse environments without further training on varied datasets.

## Acknowledgments

- This project uses OpenCV for real-time video processing and TensorFlow for deep learning model training.
- The alert sound used in this project is from [Pixabay](https://pixabay.com/sound-effects/search/alert/).

Feel free to contribute to this project by opening issues or submitting pull requests.
