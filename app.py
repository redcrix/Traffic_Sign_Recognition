from fastai.vision import *
from flask import Flask, json, request
import time
path = 'model_traffic/'
learn = load_learner(path, 'export.pkl')
learn = learn.load('stage-2')

app = Flask(__name__)


@app.route("/traffic", methods=['POST', 'GET'])
def traffic():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    if request.method == 'POST':
        print(request.files)
        if 'image' not in request.files:
            return json.dumps({"message": 'No image found', "success": False})
        image = request.files['image']
        image.save('./images/' + time_str + '_image.jpg')
        img = open_image('./images/' + time_str + '_image.jpg')
        #prediction
        pred_class, pred_idx, outputs = learn.predict(img)

        # Mapping ClassID to traffic sign names
        signs = []
        with open('label_names.txt', 'r') as csvfile:
            signnames = csv.reader(csvfile, delimiter='\t')
            next(signnames, None)
            for row in signnames:
                signs.append(row[1])
            csvfile.close()
        return json.dumps({"Index=": pred_class.obj, "Sign_name": signs[int(pred_class.obj)]})

@app.route("/")
def hello():
    return "Traffic Sign Recognition"

if __name__ == "__main__":
    app.run(debug=True)

