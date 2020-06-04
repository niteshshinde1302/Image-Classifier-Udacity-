import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import argparse
import numpy as np
import json

image_size = 224

def process_image(image):
    tensor_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    resized_image = tf.image.resize(image,(image_size,image_size))
    normalized_image = resized_image/255
    return normalized_image

def predict(image_path,model,top_k):
    loaded_model = tf.keras.models.load_model(model,custom_objects={'KerasLayer':hub.KerasLayer})
    print(loaded_model.summary())
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)    
    print(image_path.split("/")[-1])
    #plt.imshow(processed_test_image)
    processed_test_image=np.expand_dims(processed_test_image,0)
    probs=loaded_model.predict(processed_test_image)
    return tf.nn.top_k(probs, k=top_k)

def get_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    # Class names contain index from 1 to 102, whereas the datasets have label indices from 0 to 101, hence     remapping
    class_names_new = dict()
    for key in class_names:
        class_names_new[str(int(key)-1)] = class_names[key]
    return class_names_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument("--img_path",help="Image Path", default="./test_images/wild_pansy.jpg")
    parser.add_argument("--model",help="Model Path", default="./my_model.h5")
    parser.add_argument("--top_k", help="Fetch top k predictions", required = False, default = 5)
    parser.add_argument("--category_names", help="Class map json file", required = False, default = "label_map.json")
    args = parser.parse_args()
    
    all_class_names = get_class_names(args.category_names)
    print(args.img_path, args.model, args.top_k)
    probs, classes = predict(args.img_path, args.model, int(args.top_k))
    print("Probabilities: ",probs)
    print("Classes: ",classes)
#     print(all_class_names)
    class_names = []
    for c in classes[0]:
#         print(str(c.numpy()))
        class_names.append(all_class_names.get(str(c.numpy())))
    print("Class Names: ",class_names)
    class_label_probs = list(zip(class_names,probs[0].numpy()))
    print("Top_k classes with probabilities: ",class_label_probs)
    print("Label: ",class_names[0])