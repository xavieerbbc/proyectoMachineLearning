from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
# Create your views here.
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from tensorflow import Graph
from tensorflow.compat.v1 import Session
from io import BytesIO
from six.moves import urllib

img_height, img_width=224,224
with open('./models/modelo.json','r') as f:
    labelInfo=f.read()



labelInfo=json.loads(labelInfo)

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model=load_model('./models/pesos.h5')


def index(request):
    context = {'a':1}
    return render(request, 'index.html', context)

def predictImage(request):
    print(request)
    print(request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    print(filePathName)
    testimage='.'+filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x=x/255
    x=x.reshape(1,img_height, img_width,3)
    predi = None
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)

    import numpy as np
    predictedLabel=labelInfo[str(np.argmax(predi[0]))]
    print(max(predi[0]), ' --'*10)
    valor = float(max(predi[0]))*100
    porcentaje = round(valor, 2)
    context={'filePathName':filePathName,'predictedLabel':predictedLabel[1], 'porcentaje': porcentaje}
    return render(request, 'index.html', context)

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context) 

