# Code to import model and  weight file from drive for face_recognition.py
mkdir models/Facenet
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1' -O facenet_model.h5
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1e6PHRlIeayAsvRGpYUwvstklvJy-3H5B' -O facenet_weights.h5
mv facenet_weights.h5 models/Facenet
mv facenet_model.h5 models/Facenet