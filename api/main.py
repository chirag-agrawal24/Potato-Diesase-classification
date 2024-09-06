from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app=FastAPI()

model=tf.keras.models.load_model('../models/my_model_v2.h5')
class_names=['Early_blight', 'Late_blight', 'Healthy']

@app.get("/ping")
async def check_server_status():
    return "Hello , I am alive"

def read_file_as_image(data)-> np.ndarray:
    '''Data coming from user is in bytes format'''
    image=np.array(Image.open(BytesIO(data)))# convert into pillow image format then to array
    return image


@app.post("/predict")
async def predict(
    file: UploadFile):#file is variable UploadFile is datatype
    image = read_file_as_image(await file.read() )# async and await 
    img_batch=np.expand_dims(image,0)
    pred=model.predict(img_batch)[0]
    class_pred=class_names[np.argmax(pred)]
    confidence=float(np.max(pred))#np float is not allowed to return




    return {'confidence':confidence,
            'class':class_pred}
    

if __name__=="__main__":
    uvicorn.run('main:app',host="localhost",port=8000,reload=True)