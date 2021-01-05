import argparse
import json
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np



def decode_img(img):
    """Decode image and resize
    """
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [200, 300])

    return img


def process_path(file_path):
    """Process input path
    """
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, file_path


def plot_results(infer_images, inference_predicted_class, inference_predictions, class_names=['plants', 'water']):
    """Plot four images with predicted class and probabilities
    """
    for i, (infer_img, _) in enumerate(infer_images.take(4)):
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(infer_img.numpy()/255)

        # Find the predicted class from predictions
        m = "Predicted: {}, {:.2f}%".format(
            class_names[inference_predicted_class[i]], inference_predictions[i]*100)
        plt.title(m)
        plt.axis("off")
    plt.show()

def run_inference(infer_images, model_path):

    trained_model = load_model(model_path, compile=False)

    inference_predicted_class = []
    inference_predictions = []
    results = {}
    for infer_img, img_name in infer_images:
        preds = trained_model.predict(tf.expand_dims(infer_img, axis=0))
        inference_predicted_class.append(np.argmax(preds))
        inference_predictions.append(preds[0][np.argmax(preds)])

        results[str(img_name.numpy().decode('utf8').split('/')[-1])
                ] = {"class": int(np.argmax(preds)), "prob": float(preds[0][np.argmax(preds)])}

    plot_results(infer_images, inference_predicted_class,
                 inference_predictions)


    return results

def save_results(results):
    """Save results to json
    """
    json.dump(results, open("results.json", "w"))

def main(test_dir, model_path):

    # get the count of image files in the train directory
    inference_ds = tf.data.Dataset.list_files(test_dir + '/*', shuffle=False)

    infer_images = inference_ds.map(process_path)

    #inference
    results = run_inference(infer_images, model_path)
    
    #save results
    save_results(results)


if __name__ == '__main__':
    
    # Initiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', action='store', help='Dataset path')
    parser.add_argument('-model', action='store', help='Model path')
    arguments = parser.parse_args()

    dataset = arguments.data
    model = arguments.model
    main(dataset, model)
