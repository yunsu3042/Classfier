import sys
import tensorflow as tf
import tools
import numpy as np
import vgg
import argparse
from skimage.transform import resize
from skimage.io import imread

G = tf.Graph()
with G.as_default():
    images = tf.placeholder("float", [1, 224, 224, 3])
    logits = vgg.build(images, n_classes=17, training=False)
    probs = tf.nn.softmax(logits)

def predict(im):
    labels = ["뷰티", "자동차_공구", "의류", "컴퓨터", "반려동물", "취미", "식품", "건강",
              "도서_문구", "스포츠_레저", "디지털", "가구_인테리어", "가전", "생필품_주방"
               , "잡화","여행_e쿠폰", "출산_육아"]
    
    if im.shape != (224, 224, 3):
        im = resize(im, (224, 224))
    im = np.expand_dims(im, 0)
    sess = tf.get_default_session()
    results = sess.run(probs, {images: im})
    #return labels[np.argmax(results)]
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", required=True, help="path to weights.npz file")
    parser.add_argument("image", help="path to jpg image")
    args = parser.parse_args()
    im = imread(args.image)
    sess = tf.Session(graph=G)
    with sess.as_default():
        tools.load_weights(G, args.weights)
        print(predict(im))
