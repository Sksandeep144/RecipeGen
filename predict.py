from preprocess import train_images
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from IPython.display import display, HTML
import numpy as np
import pandas as pd


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# def pretty_print(df):
#     return display(HTML(df.to_html().replace("\\r\\n", "<br>")))


def predict():
    pd.set_option('display.max_colwidth', None)
    class_names = list(train_images.class_indices.keys())
    print(class_names)
    model = load_model('C:\CLASS\python\Rec\saved2.h5')
    img = load_img('static/images/image.jpg', target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images)
    text = class_names[np.argmax(classes)]
    text2 = text.upper().replace('_', ' ')
    FoodData = pd.read_csv('C:\CLASS\python\Rec\\recipie.csv', index_col=False)
    Food_ingredients_recipe = FoodData[FoodData['Food'] == text]
    # print("\n"+color.BOLD + color.UNDERLINE + color.GREEN + text.upper().replace('_', ' ') + color.END)
    # print(color.BOLD + color.UNDERLINE + color.YELLOW + "Ingredients" + color.END)
    items = Food_ingredients_recipe['Ingredients'].to_string(index=False).replace('\\r\\n', '<br><br>')
    # items = items.replace(r"\(.*\)", "")
    # print(color.BOLD + color.UNDERLINE + color.YELLOW + "Recipe" + color.END)
    recipe = Food_ingredients_recipe['Recipe'].to_string(index=False).replace('\\r\\n', '<br><br>')

    # Food_ingredients = pd.DataFrame(Food_ingredients_recipe['Ingredients'])
    # Food_Recipe = pd.DataFrame(Food_ingredients_recipe['Recipe'])
    # stuff = pretty_print(Food_ingredients)
    # things = pretty_print(Food_Recipe)
    return text2, items, recipe
