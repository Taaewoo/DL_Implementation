import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import Net
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from PIL import Image
from matplotlib import pyplot
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


face = extract_face("Joo.jpg")

pyplot.axis('off')
#pyplot.imshow(face)
#pyplot.show()

face = [face]

face = np.array(face)

face = np.transpose(face,(0,3,1,2))

face = torch.from_numpy(face)


print(face.shape)
net = Net()
#print(net)
outputs = net(face.float())
print(outputs)