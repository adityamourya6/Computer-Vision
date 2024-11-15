import numpy as np
import os
import glob
import cv2
import onnxruntime
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

img = cv2.imread("3.jpg")
plt.imshow(img[:,:,::-1])
plt.show()

faces = app.get(img)
print(len(faces))


fig,axs=plt.subplots(1, len(faces), figsize=(12,5))

for i, face in enumerate (faces):
    bbox = face['bbox']
    bbox = [int(b) for b in bbox]
    axs[i].imshow(img[bbox[1]:bbox[3],bbox[0]: bbox[2],::-1])
    axs[i].axis('off')
    plt.show()
model_path = "inswapper_128.onnx"
swapper = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)

source_face = faces [0]
bbox = source_face['bbox']
bbox = [int(b) for b in bbox]
plt.imshow(img[bbox[1]:bbox [3], bbox[0]:bbox[2],::-1])
plt.show()

res = []

for face in faces:
    _img,_ = swapper.get(img, face, source_face, paste_back=False)
    res.append(_img)
res = np.concatenate(res, axis=1)
fig,ax = plt.subplots(figsize=(15,5))
ax.imshow(res[:,:,::-1])
ax.axis('off')
plt.show()

rob = cv2.imread('varma1.jpg')
rob_faces=app.get(rob)
rob_face = rob_faces[0]
#Replace faces in friends image
res = img.copy()
for face in faces:
    res = swapper.get(res, face, rob_face, paste_back=True)
fig, ax = plt.subplots()
ax.imshow(res[:,:,::-1])
ax.axis('off')
plt.show()