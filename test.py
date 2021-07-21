# importamos librerías.
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

net = cv2.dnn.readNet(prototxtPath,weightsPath)

# Cargamos una imagen y predecimos si con la red de deteccion de caras.
image_pred = cv2.imread('/home/julio/Descargas/mj1.png')
blob = cv2.dnn.blobFromImage(image_pred,1.0,(300,300))#imagen, scale_factor, size of image.
(h,w) = image_pred.shape[:2]

net.setInput(blob)
detections = net.forward()#like predict function.


# si hemos detectado caras, predecimos con nuestra red de deteccion de mascarilla para cada cara.
for i in range(detections.shape[2]):
    prob = detections[0,0,i,2]
    #print(prob)
    if prob >= 0.15:
        
        # nos quedamos solo con el box de la cara.
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        print(box)
        (startX, startY, endX, endY) = box.astype("int")
       
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        face = image_pred[startY:endY,startX:endX]
        #cv2.imshow("ventana",face)
        #cv2.waitKey(3000)

        
        # preparamos la imagen de la cara para poder pasarla a nuestra red.
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)#formato pil
        face = cv2.resize(face,(150,150))
        face = face/255.
        face = np.expand_dims(face,axis=0)

        # predecimos
        pred = model.predict(face)
        #0 con mascarilla, 1 sin.

        pred = round(pred[0][0],3)
        
        # establecemos un umbral para la funcion de activacion sigmoide en 0.5 y mostramos en la imagen.
        if pred >= 0.5:
            cv2.putText(image_pred,'No mask {}'.format(str(pred)),(startX,startY),color=(0,0,255),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, thickness=1 )    
            cv2.rectangle(image_pred,(startX,startY),
                    (endX,endY),(0,0,255))
        
        else:
            cv2.putText(image_pred,'Mask {} '.format(str(1-pred)),(startX,startY),color=(0,255,0),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, thickness=1 )    
            cv2.rectangle(image_pred,(startX,startY),
                    (endX,endY),(0,255,0))
        
cv2.imshow("ventana",image_pred)
cv2.waitKey(10000)


# codigo para obtener accuracy en el conjunto de test.
'''
i = 0
pred = []
labels = []
for img, label in test_datagen:

   
    pred_aux = model.predict(img)
    pred_aux = pred_aux[0,0]
    pred_aux = round(pred_aux)
    pred_aux = int(pred_aux)
    print(pred_aux)
    labels.append(label)
    pred.append(pred_aux)
    i=i+1
    print(i)
    if i >=1000:
        break

cont=0

print(pred)
print(label)
for i in range(len(pred)):
    if pred[i] == labels[i]:
        cont += 1

print(cont/len(pred))
#0.986 de accuracy en test.
'''