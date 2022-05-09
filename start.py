from flask import Flask, render_template, request
from predict import predict

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save('static/images/image.jpg')
        food, item, recipe = predict()
        #img_path = 'static/image.jpg'
        # return render_template("wait.html", abc=b)
    return render_template("success.html", food=food, item=item, recipe=recipe)


if __name__ == "__main__":
    app.run(debug=True)
