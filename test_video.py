# importamos librerias
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from keras.preprocessing import image

import os
import matplotlib.pyplot as plt
import numpy as np

import cv2
from PIL import Image
from keras.models import model_from_json


# Cargamos datos 

#model = models.load_model('masks.h5')
base_dir = '/home/julio/Escritorio/JULIO/Mask_project/archive/FaceMaskDataset'

train_dir = os.path.join(base_dir,'Train')
validation_dir = os.path.join(base_dir,'Validation')
test_dir = os.path.join(base_dir,'Test')

datagen = ImageDataGenerator(rescale=1./255)

train_datagen = datagen.flow_from_directory(train_dir,
                                            target_size=(150,150),
                                            batch_size=32,class_mode='binary')

validation_datagen = datagen.flow_from_directory(validation_dir,
                                            target_size=(150,150),
                                            batch_size=32,class_mode='binary')

test_datagen = datagen.flow_from_directory(test_dir,
                                            target_size=(150,150),
                                            batch_size=1,class_mode='binary')


# Cargamos el mejor modelo.

print("Cargando modelo...")
# load json and create model
json_file = open('models/model-{epoch:02d}-{val_accuracy:02d}.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("models/model-{epoch:02d}-{val_accuracy:02d}.h5")
print("Loaded model from disk")


# Cargamos la red de opencv entrenada para detectar caras.
prototxtPath = '/home/julio/Escritorio/JULIO/Mask_project/facedetector/deploy.prototxt'
weightsPath = '/home/julio/Escritorio/JULIO/Mask_project/facedetector/res10_300x300_ssd_iter_140000.caffemodel'

# Cargamos una imagen y predecimos si con la red de deteccion de caras.
net = cv2.dnn.readNet(prototxtPath,weightsPath)

# creamos los objetos para el video
cap = cv2.VideoCapture(0)

# bucle que esta captando frames de la camara.
while True:

    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame,1.0,(300,300))#imagen, scale_factor, size of image.
    (h,w) = frame.shape[:2]

    net.setInput(blob)
    detections = net.forward()#like predict function


    for i in range(detections.shape[2]):
        prob = detections[0,0,i,2]
        #print(prob)
        if prob >= 0.5:

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            print(box)
            (startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY,startX:endX]
            #cv2.imshow("ventana",face)
            #cv2.waitKey(3000)

            

            face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)#formato pil
            face = cv2.resize(face,(150,150))
            face = face/255.
            face = np.expand_dims(face,axis=0)

            pred = model.predict(face)
    #0 con mascarilla, 1 sin.

            pred = round(pred[0][0],3)
            #print(pred)
            if pred >= 0.5:
                cv2.putText(frame,'No mask {}'.format(str(pred)),(startX,startY),color=(0,0,255),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, thickness=1 )    
                cv2.rectangle(frame,(startX,startY),
                        (endX,endY),(0,0,255))
            
            else:
                cv2.putText(frame,'Mask {} '.format(str(1-pred)),(startX,startY),color=(0,255,0),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, thickness=1 )    
                cv2.rectangle(frame,(startX,startY),
                        (endX,endY),(0,255,0))

        cv2.imshow("ventana",frame)
        #cv2.waitKey(1000)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break    


cap.release()
cv2.destroyAllWindows()
