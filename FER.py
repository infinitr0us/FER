import cv2
import numpy as np
import tensorflow as tf
import time

from Training import image_to_tensor, deepnn

global time_f
global time_l
global emoji_face

time_f=time.time()

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'fearful', 'happy', 'sad', 'surprised']

def formation(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(image,scaleFactor = 1.3,minNeighbors = 5)

  # None is no face found in image
  if not len(faces) > 0:
    return None, None
  max_are_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
      max_are_face = face

  # face to image
  face_coor =max_are_face
  image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]

  # Resize image to network size
  try:
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC) #立方插值
  except Exception:
    print("[+} Problem during resize")
    return None, None
  return  image, face_coor

def demo(modelPath='./models', showBox=False):
  face_x = tf.placeholder(tf.float32, [None, 2304])
  # Build the graph for the deep net
  y_conv = deepnn(face_x)
  # 调用softmax回归，进行分类
  probs = tf.nn.softmax(y_conv)

  # 读取训练好的网络
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(modelPath)

  sess = tf.Session()
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restore  sucssesful!!')

  feelings_faces = []
  for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./data/emojis/' + emotion + '.png', -1))

  video_captor = cv2.VideoCapture(0)

  global time_f
  global time_l
  global emoji_face

  emoji_face = []
  result = None

  while True:
    ret, frame = video_captor.read()
    detected_face, face_coor = formation(frame)
    if showBox:
      if face_coor is not None:
        [x,y,w,h] = face_coor
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    if 1==1:
      if detected_face is not None:
        tensor = image_to_tensor(detected_face)
        result = sess.run(probs, feed_dict={face_x: tensor})
    if result is not None:
      for index, emotion in enumerate(EMOTIONS):
        if face_coor is not None:
          time_l=time.time()
          if (time_l-time_f>1):
            time_f=time_l
            emoji_face = feelings_faces[np.argmax(result[0])]

          [x, y, w, h] = face_coor
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for c in range(0, 3):
          frame[10:130, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 2] / 255.0) + frame[10:130, 10:130, c] * (1.0 - emoji_face[:, :, 2] / 255.0)
    cv2.imshow('face', frame)
    if cv2.waitKey(1)  == ord('q'):
      break

if __name__ == '__main__':
  demo('./models')