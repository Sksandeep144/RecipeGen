import pickle
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import test_images
import matplotlib.pyplot as plt
import seaborn as sns
import json

model = load_model('C:\CLASS\python\Rec\saved2.h5')
# history_dict = json.load(open('histories', 'r'))
history = pickle.load(open('histories.pkl', "rb"))
results = model.evaluate(test_images, verbose=0)
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
predictions = np.argmax(model.predict(test_images), axis=1)
cm = confusion_matrix(test_images.labels, predictions)
clr = classification_report(test_images.labels, predictions, target_names=test_images.class_indices, zero_division=0)
plt.figure(figsize=(30, 30))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks=np.arange(len(test_images.class_indices.keys())) + 0.5, labels=test_images.class_indices, rotation=90)
plt.yticks(ticks=np.arange(len(test_images.class_indices.keys())) + 0.5, labels=test_images.class_indices, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print("Classification Report:\n----------------------\n", clr)
# N = 50
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(history['accuracy'])
# plt.plot(history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
# plt.savefig("CNN_Model")
