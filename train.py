# importamos librerías
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
import keras
import os
import matplotlib.pyplot as plt
from keras.models import model_from_json


# Creamos una clase que nos permite guardar el modelo y los pesos en cada iteracion.
class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # serialize model to JSON
        self.model_json = model.to_json()
        with open("models/model-{epoch:02d}-{val_accuracy:02d}.json", "w") as json_file:
            json_file.write(self.model_json)
        # serialize weights to HDF5
        self.model.save_weights("models/model-{epoch:02d}-{val_accuracy:02d}.h5")
        print("Saved model to disk")

# path del dataset en mi pc local.
base_dir = '/home/julio/Escritorio/JULIO/Mask_project/archive/FaceMaskDataset'

# path a cada conjunto: train, validacion y test.
train_dir = os.path.join(base_dir,'Train')
validation_dir = os.path.join(base_dir,'Validation')
test_dir = os.path.join(base_dir,'Test')

# Creamos un objeto de ImageDataGenerator que va a aplicar data augmentation
datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,
                                    height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,
                                    fill_mode='nearest')

# Cargamos los datos de train.
train_datagen = datagen.flow_from_directory(train_dir,
                                            target_size=(150,150),
                                            batch_size=32,class_mode='binary')

# cargamos los datos de test.
validation_datagen = datagen.flow_from_directory(validation_dir,
                                            target_size=(150,150),
                                            batch_size=32,class_mode='binary')

# pequeño codigo para mostrar imagenes.
'''
i=0
for img, label in train_datagen:

    print(label)
    plt.imshow(img[0])
    plt.show()

    i += 1

    if i >= 10:
        break

'''

# Cargamos la red VGG19 con los pesos de imagenet, sin incluir la ultima capa, esto lo indicamos en include_top = False.
conv_base = VGG19(weights='imagenet',input_shape=(150,150,3),include_top=False)

# Creamos un modelo secuencial.
model = models.Sequential()

# A la red VGG19 le aplicamos Flatten para conventir las caracteristicas en un vector, aplicamos dropout
# para ayudar a la red a generalizar y añadimos una capa de 512 neuronas y la capa final que clasifica entre
# mascarilla o no mascarilla.
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

# aqui indicamos que el modelo de VGG19 que hemos cargado queremos que no se entrene, que 
# solo se entrene las capas añadidas.
conv_base.trainable = False

# mostramos un resumen de la red.
model.summary()

# compilamos el modelo.
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])

# creamos el objeto que nos guarda en cada epoca.
saver = CustomSaver()

# entrenamos el modelo.
history = model.fit_generator(train_datagen,validation_data=validation_datagen,epochs=30,steps_per_epoch=100,validation_steps=25,callbacks=[saver])

#model.save('masks.h5')


# obtenemos los resultados del entrenamiento y los mostramos.
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

#creamos otra figura
plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()