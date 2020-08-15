# FaceNet - PyTorch  
## Week 1  
- TODO :  FaceNet Paper Review.  
- Paper : [FaceNet](https://arxiv.org/pdf/1503.03832.pdf)  
- Review : [PDF Link](https://github.com/Taaewoo/Paper_review/blob/master/09.%20FaceNet%3B%20A%20Unified%20Embedding%20for%20Face%20Recognition%20and%20Clustering.pdf)  
  
## Week 2  
- TODO : FaceNet Code Review & implementation using PyTorch ( extracting 128-d feature vector, calculating squared L2 distance )
- Reference : [FaceNet-tensorflow github](https://github.com/davidsandberg/facenet)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[FaceNet-PyTorch github](https://github.com/tbmoon/facenet)  
<br>  

### NN
<img src="https://user-images.githubusercontent.com/28804154/88418802-12075580-ce1f-11ea-85e8-b1974475b593.png"  width="80%" height="80%">

<br>  

### **MTCNN result of triplet**
<img src="https://user-images.githubusercontent.com/28804154/88461635-265f5700-cee0-11ea-8ed7-52f073ce5a33.png"  width="50%" height="50%">

<br>  

### **Distance of triplet**
![image](https://user-images.githubusercontent.com/28804154/88461780-87d3f580-cee1-11ea-8a4d-bd832cc3d82b.png)
  
<br>  
  
## Week 3
- TODO : Cropping CASIA-WebFace dataset using MTCNN & Customizing the PyTorch Dataset class for triplet.
- CASIA-WebFace dataset : [Download link](https://drive.google.com/open?id=1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz)
<br>  

### **Crop the CASIA-WebFace image files using MTCNN**
#### - It takes about 7 hours to crop 491,542 image files
#### - I used [facenet github's align soure code](https://github.com/davidsandberg/facenet/tree/master/src/align)
<img src="https://user-images.githubusercontent.com/28804154/89706169-1ba3c800-d99e-11ea-87b2-ab10edc72ffa.png"  width="50%" height="50%">

<br>

### **Override PyTorch Dataset class method for custom triplet data**
#### - Customizing dataset
~~~python
class TripletDataset(torch.utils.data.Dataset): 
  def __init__(self):
      # override
      # Initializing class variable like dataset path, pytorch transform, custom data.
      
  def __len__(self):
      # override
      # return custom data length.

  def __getitem__(self, idx): 
      # override
      # return custom data with index.
      
  def generate_triplet(self):
      # generate triplet set using CASIA-WebFace dataset.
~~~
<br>

#### - Load the custom data as much as the batch size
~~~python
trans = transforms.Compose([transforms.ToTensor()])

trainloader = DataLoader(dataset=TripletDataset(root_dir="data/CASIA-WebFace-MTCNN", transform = trans),
                         batch_size = 64, 
                         num_workers=20)
                         
for i, data in enumerate(trainloader):
    # batch number i
    # bath size data
~~~
<br>

## Week 4
- TODO : Implementation of selecting triplet from CASIA-WebFace dataset.
<br>

### **Load triplet set using DataLoader & Calculate L2 distance**
<img src="https://user-images.githubusercontent.com/28804154/89710067-e6f23980-d9ba-11ea-86e7-827486fbd14e.png"  width="50%" height="50%">
<br>

### **Selecting semi-hard dataset & Calculate triplet loss**
~~~python
pos_dis = l2_distance.forward(anc_fv,pos_fv)
neg_dis = l2_distance.forward(anc_fv,neg_fv)

all = (pos_dis < neg_dis).cpu().numpy().flatten()

losses = torch.clamp(pos_dis[all] - neg_dis[all] + margin, min=0.0)
loss = torch.mean(losses)
~~~
<br>
  
## Week 5
- TODO : Learning neural network & Validation.
<br>

### **Transfer tensor variable CPU to GPU**
#### - It is essential to use GPU for PyTorch neural network learning
~~~python
model.cuda()

anc_fv = model(data['anc_img'].cuda())
pos_fv = model(data['pos_img'].cuda())
neg_fv = model(data['neg_img'].cuda())
~~~
<br>

### **Learning neural network**
<img src="https://user-images.githubusercontent.com/28804154/89985202-a42faa80-dcb5-11ea-8ec9-8fb9bbd963cb.png"  width="50%" height="50%">
<br>

### **Prepare validation dataset**
<img src="https://user-images.githubusercontent.com/28804154/89986048-f1f8e280-dcb6-11ea-8139-cb23215ecfb9.png"  width="50%" height="50%">
<br>

### **Training result**
<img alt="image" src="https://user-images.githubusercontent.com/28804154/90315668-27702b00-df58-11ea-86dd-ff43d26620e7.png" width="80%" height="80%">
<img alt="image" src="https://user-images.githubusercontent.com/28804154/90315753-e2002d80-df58-11ea-96e6-18e1b777af3f.png" width="80%" height="80%">
<img alt="image" src="https://user-images.githubusercontent.com/28804154/90322974-fdd5f480-df95-11ea-88dd-3f54e902ca8f.png" width="80%" height="80%">

