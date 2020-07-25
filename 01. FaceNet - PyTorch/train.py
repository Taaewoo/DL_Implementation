import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from model import Net
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from PIL import Image
from torchvision.models import resnet50

# 주어진 사진에서 하나의 얼굴 추출
def extract_face(filename, required_size=(220, 220)):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = asarray(image)
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# 220:220:3
anchor = extract_face("Joo.jpg")
positive = extract_face("Joo2.jpg")
negative = extract_face("Kim.jpg")

face = [anchor, positive, negative]


fig = plt.figure()

ax1 = fig.add_subplot(1,3,1)
ax1.imshow(anchor)
ax1.set_title("Anchor")
ax1.axis("off")

ax1 = fig.add_subplot(1,3,2)
ax1.imshow(positive)
ax1.set_title("Positive")
ax1.axis("off")

ax1 = fig.add_subplot(1,3,3)
ax1.imshow(negative)
ax1.set_title("Negative")
ax1.axis("off")

plt.show()


face = np.array(face)
face = np.transpose(face,(0,3,1,2))
face = torch.from_numpy(face)

print(face.shape)
net = Net()
#print(net)
#print(face.float())
outputs = net(face.float())
#print(outputs)
print("")
print("Anchor <-> Positive distance : " + str(((outputs[0] - outputs[1])**2).sum().item()) )
print("Anchor <-> Negative distance : " + str(((outputs[0] - outputs[2])**2).sum().item()) )

