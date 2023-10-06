import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, Response

app = Flask(__name__)
nama = ""

cap = cv2.VideoCapture(1)


# face recognition
def show_dataset(images_class, label):
    # show data for 1 class
    plt.figure(figsize=(14, 5))
    k = 0
    for i in range(1, 6):
        plt.subplot(1, 5, i)
        try:
            plt.imshow(images_class[k][:, :, ::-1])
        except:
            plt.imshow(images_class[k], cmap='gray')
        plt.title(label)
        plt.axis('off')
        plt.tight_layout()
        k += 1
    plt.show()


dataset_folder = "dataset/"
names = []
images = []
for folder in os.listdir(dataset_folder):
    # limit only 70 face per class
    for name in os.listdir(os.path.join(dataset_folder, folder))[:70]:
        img = cv2.imread(os.path.join(dataset_folder + folder, name))
        images.append(img)
        names.append(folder)
labels = np.unique(names)

for label in labels:
    ids = np.where(label == np.array(names))[0]
    images_class = images[ids[0]: ids[-1] + 1]

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')


def detect_face(img, idx):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    try:
        x, y, w, h = faces[0]

        img = img[y:y + h, x:x + w]
        img = cv2.resize(img, (100, 100))
    except:
        img = None
    return img


new_names = []
croped_images = []

for i, img in enumerate(images):
    img = detect_face(img, i)
    if img is not None:
        croped_images.append(img)
        new_names.append(names[i])

names = new_names

for label in labels:
    ids = np.where(label == np.array(names))[0]
    # select croped images for each class
    images_class = croped_images[ids[0]: ids[-1] + 1]

name_vec = np.array([np.where(name == labels)[0][0] for name in names])

model = cv2.face.LBPHFaceRecognizer_create()
model.train(croped_images, name_vec)
model.save("lbph_model.yml")
model.read("lbph_model.yml")


def draw_ped(img, label, x0, y0, xt, yt, color=(255, 127, 0), text_color=(255, 255, 255)):
    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img,
                  (x0, y0 + baseline),
                  (max(xt, x0 + w), yt),
                  color,
                  2)
    cv2.rectangle(img,
                  (x0, y0 - h),
                  (x0 + w, y0 + baseline),
                  color,
                  -1)
    cv2.putText(img,
                label,
                (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA)
    return img


def generate_face_frames():
    global nama
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (100, 100))

                idx, confidence = model.predict(face_img)

            if 0 <= idx < len(labels):
                label_text = "%s (%.2f %%)" % (labels[idx], confidence)
                nama = labels[idx]

                frame = draw_ped(frame, label_text, x, y, x + w, y + h,
                                 color=(0, 255, 255), text_color=(50, 50, 50))
            else:
                nama = ""
        else:
            nama = ""

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# website konfigurasi
@app.route('/')
def index():
    return render_template('index.php')


@app.route('/video_feed_face')
def video_feed_face():
    return Response(generate_face_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_nama')
def get_nama():
    global nama
    return nama


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
