#!/usr/bin/env python
# coding: utf-8

# 

# # 2. CIFAR10 데이터셋 학습 모델 만들기

# ## 2-0. Library

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device : gpu") if torch.cuda.is_available() else print("device : cpu")


# ## 2-1. Hyper-Parameter Setting

# In[2]:


learning_rate = 1e-1
batch_size = 128
dropout_rate = 0.2

best_acc = 0
start_epoch = 0 # start from epoch 0 or last checkpoint epoch

activation = nn.ReLU()
max_pool = nn.MaxPool2d(2,2)


# ## 2-2. Load Data & Data Pre-processing

# In[3]:


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last = True)

test_data = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False, num_workers=2, drop_last = True)


# data 확인
x_train, y_train = train_data[0]
x_train = np.transpose(x_train, (1 , 2, 0))

print("data :", x_train)
print("data shape :", x_train.shape)
print("label :", y_train)

plt.figure()
plt.imshow(x_train)
plt.show()


examples = enumerate(train_loader)
batch_idx, (example_data, example_target) = next(examples)

print('data shape:', example_data.shape)
print('label:', example_target)

check_image = example_data[0]
check_image = np.transpose(check_image, (1, 2, 0))

plt.figure()
plt.imshow(check_image)
plt.show()


# ## 2-3. Model & Opimizatino and Loss function

# In[4]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() # for initializing nn.Module (parent class)
        
        # 1. CNN : feature 추출
        self.feature_extraction = nn.Sequential(
            # 입력 : 3 * 32 * 32
            
            nn.Conv2d(3, 64, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(64),
            activation, 
            #nn.Dropout2d(dropout_rate),
            # 64 * 32 * 32
            nn.Conv2d(64, 64, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(64),
            activation, 
            #nn.Dropout2d(dropout_rate),
            # 64 * 32 * 32
            max_pool,
            # 64 * 16 * 16
            
            
            nn.Conv2d(64, 128, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(128),
            activation,
            #nn.Dropout2d(dropout_rate),
            # 128 * 16 * 16
            nn.Conv2d(128, 128, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(128),
            activation,
            #nn.Dropout2d(dropout_rate),
            # 128 * 16 * 16
            max_pool,
            # 128 * 8 * 8
            
            
            nn.Conv2d(128, 256, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(256),
            activation,
            #nn.Dropout2d(dropout_rate),
            # 256 * 8 * 8
            nn.Conv2d(256, 256, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(256),
            activation,
            #nn.Dropout2d(dropout_rate),
            # 256 * 8 * 8
            nn.Conv2d(256, 256, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(256),
            activation,
            #nn.Dropout2d(dropout_rate),
            # 256 * 8 * 8
            max_pool,
            # 256 * 4 * 4
            
            
            nn.Conv2d(256, 512, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(512),
            activation,
            #nn.Dropout2d(dropout_rate),
            # 512 * 4 * 4
            nn.Conv2d(512, 512, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(512),
            activation,
            #nn.Dropout2d(dropout_rate),
            # 512 * 4 * 4
            nn.Conv2d(512, 512, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(512),
            activation,
            #nn.Dropout2d(dropout_rate),
            # 512 * 4 * 4
            max_pool,
            # 512 * 2 * 2
            
            
            nn.Conv2d(512, 512, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(512),
            activation,
            #nn.Dropout2d(dropout_rate),
            # 512 * 2 * 2
            nn.Conv2d(512, 512, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(512),
            activation,
            #nn.Dropout2d(dropout_rate),
            # 512 * 2 * 2
            nn.Conv2d(512, 512, 3, padding = 1), # kernel_size = 3에 padding = 1을 두면 사이즈가 유지됨
            nn.BatchNorm2d(512),
            activation,
            #nn.Dropout2d(dropout_rate),
            # 512 * 2 * 2
            max_pool,
            # 512 * 1 * 1
            
            nn.AvgPool2d(1, 1)
            # 512 * 1 * 1
        )
        
        
        # 2. CNN : 분류
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 10),
            #activation,
            #nn.Dropout(dropout_rate),
        )

        
    # 학습
    def forward(self, x):
        # 입력 : 3 * 32 * 32
        
        extracted_feature = self.feature_extraction(x)
        #print(extracted_feature.shape)
        flatten = extracted_feature.view(batch_size, -1)
        result = self.classifier(flatten)
        return result

    
model = CNN().to(device)
#model.train()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)


# ## 2-4. Train & Test

# In[5]:


# Training

train_loss_array = []

def train(epoch):
    print("\nEpoch :", epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, [data, label] in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
                
        optimizer.zero_grad()
        output = model.forward(data)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, prediction_index = torch.max(output, 1)
        total += label.size(0)
        correct += (prediction_index == label).sum().float()
        
        print("\t%d / %d" %(batch_idx, len(train_loader)))
        print("\ttrain loss : %.3f | Acc : %3f%% (%d / %d)" %(train_loss/(batch_idx+1), correct/total*100, correct, total))
        
        train_loss_array.append(loss.cpu().detach().numpy())


# In[6]:


#test the model

test_loss_array = []

def test(epoch):
    global best_acc
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, [data, label] in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model.forward(data)
            loss = loss_function(output, label)
            
            test_loss += loss.item()
            _, prediction_index = torch.max(output, 1)
            total += label.size(0)
            correct += (prediction_index == label).sum().float()
        
            print("\t%d / %d" %(batch_idx, len(test_loader)))
            print("\ttest loss : %.3f | Acc : %3f%% (%d / %d)" %(test_loss/(batch_idx+1), correct/total*100, correct, total))
        
            test_loss_array.append(loss.cpu().detach().numpy())
            
    acc = correct/total * 100
    if acc > best_acc:
        print("\t...Saving model...")
        state = {
            "model" : model.state_dict(),
            "acc" : acc,
            "epoch" : epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/best_model.pth")
        
        best_acc = acc


# In[7]:


for epoch in range(start_epoch, start_epoch + 150):
    train(epoch)
    test(epoch)


# In[8]:


def load_checkpoint():
    global best_acc
    global start_epoch
    
    assert os.path.isdir("checkpoint"), "Error : no checkpoint dir found!"
    
    checkpoint = torch.load("./checkpoint/best_model.pth")
    model.load_state_dict(checkpoint["model"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    
    print("best_acc :", best_acc)
    print("start_epoch :", start_epoch)


# In[9]:


load_checkpoint()


# ## 2-4-1. 모델 성능 확인
# - epoch 146에 test accuracy 85.2364
# - learning rate를 0.01로 줄여 다시 학습
# > - 빠른 결과 확인을 위해 50 epoch 수행

# In[10]:


learning_rate = 1e-2

for epoch in range(start_epoch, start_epoch + 50):
    train(epoch)
    test(epoch)


# In[11]:


load_checkpoint()


# ## 2-4-2. 모델 성능 확인
# - epoch 165에 test accuracy 85.4668
# - learning rate를 0.01로 다시 학습
# > - 빠른 결과 확인을 위해 30 epoch 수행

# In[12]:


learning_rate = 1e-2

for epoch in range(start_epoch, start_epoch + 30):
    train(epoch)
    test(epoch)


# In[13]:


load_checkpoint()


# ## 2-4-3. 모델 성능 확인
# - epoch 165에 test accuracy 85.4668로 변함이 없음
# - learning rate를 0.01로 다시 학습
# > - 빠른 결과 확인을 위해 50 epoch 수행

# In[ ]:


learning_rate = 1e-2

for epoch in range(start_epoch, start_epoch + 30):
    train(epoch)
    test(epoch)


# In[ ]:




