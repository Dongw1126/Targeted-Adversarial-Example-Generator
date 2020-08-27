import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


model = tf.keras.applications.MobileNetV2(weights='imagenet')
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
loss_function = tf.keras.losses.CategoricalCrossentropy()

input_filename = "input.jpg"
output_filename  = "output.jpg"
target_index = 333 # ImageNet class index, 333 : hamster


def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]


def get_label_shape():
  tmp = np.full((224, 224, 3), 0.5)
  out = model.predict(tmp[None, ...])
  label_shape = out.shape[-1]

  return label_shape


def preprocess_image(filename):
  image_raw = tf.io.read_file(filename)

  image = tf.image.decode_image(image_raw)
  image = tf.cast(image, tf.float32)
  image = image / 255
  image = tf.image.resize(image, (224, 224))

  return image


def get_gradients(image, labels):
  with tf.GradientTape() as tape:
    tape.watch(image)
    input_image = image[None, ...]
    prediction = model(input_image)
    loss = loss_function(labels ,prediction)

  gradients = tape.gradient(loss, image)

  return gradients


def creat_adversarial_example(_target):
  alpha = 1
  epsilon = 2.0 / 255
  steps = 10
  target = _target  # target index in ImageNet
  perturbation = np.full((224, 224, 3), 0.5)

  label_shape = get_label_shape()
  image_init = preprocess_image(input_filename)
  plt.imsave(output_filename, image_init.numpy())

  y_LL = tf.one_hot(target, label_shape)
  y_LL = tf.reshape(y_LL, (1, label_shape))

  for i in range(3):
    image_var = tf.Variable(preprocess_image(output_filename))
    x_adv = image_var

    for i in range(steps):
      gradients = get_gradients(x_adv, y_LL)

      perturbation += gradients
      below = x_adv - epsilon
      above = x_adv + epsilon
      x_adv = x_adv - alpha * tf.sign(gradients)
      x_adv = tf.clip_by_value(tf.clip_by_value(x_adv, below, above), 0, 1)

    plt.imsave(output_filename, x_adv.numpy())

  fig = plt.figure()
  panel1 = fig.add_subplot(1, 3, 1)
  _, label, confidence = get_imagenet_label(model.predict(image_init[None, ...]))
  panel1.set_title('{} \n\n {:.2f}% Confidence'.format(label, confidence * 100))
  panel1.imshow(image_init.numpy())
  panel1.axis("off")

  panel2 = fig.add_subplot(1, 3, 2)
  _, label, confidence = get_imagenet_label(model.predict(perturbation[None, ...]))
  panel2.set_title('{} \n\n {:.2f}% Confidence'.format(label, confidence * 100))
  panel2.imshow(perturbation.numpy())
  panel2.axis("off")

  panel3 = fig.add_subplot(1, 3, 3)
  _, label, confidence = get_imagenet_label(model.predict(x_adv[None, ...]))
  panel3.set_title('{} \n\n {:.2f}% Confidence'.format(label, confidence * 100))
  panel3.imshow(x_adv.numpy())
  panel3.axis("off")

  plt.show()

def image_detection(filename):
  image = preprocess_image(filename)
  out = model.predict(image[None, ...])

  plt.figure()
  plt.imshow(image.numpy())
  _, image_class, class_confidence = get_imagenet_label(out)
  plt.title('{}\n{} : {:.2f}% Confidence'.format(filename, image_class, class_confidence * 100))
  plt.show()

if __name__ == '__main__':
  creat_adversarial_example(target_index)
  image_detection(output_filename)
