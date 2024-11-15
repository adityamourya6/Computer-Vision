import numpy as np
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis

# Initialize face analysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Load images
img = cv2.imread("m6.jpg")
rob = cv2.imread("s1.jpg")

# Display original image
plt.imshow(img[:, :, ::-1])
# plt.show()

# Detect faces in images
faces = app.get(img)
rob_faces = app.get(rob)

print(len(faces))
# Choose the first detected face in 'me.jpg' as the source
rob_face = rob_faces[0]

# Select the specific face in 'Picture1.jpg' by index (e.g., the first face)
# You can change the index to specify a different face in 'faces'
target_face_index = 0  # Change this index if needed

target_face = faces[8]

# Load the face swap model
model_path = "inswapper_128.onnx"
swapper = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)
# Swap the target face in 'img' with 'rob_face'
res = img.copy()
res = swapper.get(res, target_face, rob_face, paste_back=True)

# Display the result
plt.imshow(res[:, :, ::-1])
plt.axis('off')
plt.show()
