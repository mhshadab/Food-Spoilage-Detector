The file contains:
Training image folder: this has 3 folders named Good, Strarting to Spoil and Rotten. This is where we put our training images for each classification.
SafeFood_model_trainer.py trains a deep learning model to classify images based on spoilage level using TensorFlow and Keras. 
                          It employs transfer learning with the MobileNetV2 model and data augmentation techniques to improve generalization.
                          The trained model is then saved for later use or deployment.
                          It takes the images from the trainingImages folder.
The trained model is "veggie_spoilage_predictor.h5"

SafeFood_Testerapp.py runs a PyQt5 application that uses a TensorFlow model to classify images of food as "Good," "Starting to spoil," or "Rotten."
                      Users can load and display images, and the app provides real-time predictions based on the model.
