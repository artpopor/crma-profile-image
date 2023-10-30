from flask import Flask, jsonify,request
import base64
from deepface import DeepFace
import os
from skimage.transform import resize
from skimage.io import imread
import glob
import shutil
from PIL import Image
import numpy as np
import os
import pickle
from io import BytesIO
model=pickle.load(open('crma_classify_profile.p','rb'))
Categories=['true','false']
app = Flask(__name__)

def predict(dataurl):
    base64_data = dataurl.split(',')[1]
    image_data = base64.b64decode(base64_data)
    image_np = np.array(Image.open(BytesIO(image_data)))
    # img=imread(dataurl)
    img_resize=resize(image_np,(150,150,3))
    l=[img_resize.flatten()]
    probability=model.predict_proba(l)
    truePercent = float(probability[0][0])*100
    if probability[0][0] > 0.5:
        try :
            gender = DeepFace.analyze(image_np,actions=('gender'), silent = True)
            manPercent = float(gender[0]['gender']['Man'])
        except :
            manPercent = 0
            print("Test Error!")
        return True,truePercent,manPercent
    else :
        return False,truePercent,0
    
@app.route('/', methods=['GET'])
def greeting():
    return jsonify({'message': 'process on'})

@app.route('/filterImage', methods=['POST'])
def filter_image():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data found in the payload'})
    try:
        ImageBase64 = data['image']
        status,truePercent,manPercent = predict(ImageBase64)
        return jsonify({"status":status,"truePercent":format(truePercent, ".2f"),"manPercent":format(manPercent, ".2f")})
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True, port=6000)