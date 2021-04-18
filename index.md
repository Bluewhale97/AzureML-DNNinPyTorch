# Introduction

Classical machine learning relies on statsitics to determine relationships between features and labels. In the 4Vs age of big data, there is a technique that mimic human brain to process information, every one of us knows its name as ANN, artificial neural network.

PyTorch is a framework which is convenient to perfrom DNNs. We will create a simple neural network that classifies penguins into species based on the length and depth of their culmen, flipper length and body mass.

# EDA of penguin dataset

The penguins dataset used in the this exercise is a subset of data collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER. 

Now let's load the penguin data we need. And we would implememnt some scaling for each feature.

```python
import pandas as pd

# load the training dataset (excluding rows with null values)
penguins = pd.read_csv('data/penguins.csv').dropna()

# Deep Learning models work best when features are on similar scales
# In a real solution, we'd implement some custom normalization for each feature, but to keep things simple
# we'll just rescale the FlipperLength and BodyMass so they're on a similar scale to the bill measurements
penguins['FlipperLength'] = penguins['FlipperLength']/10
penguins['BodyMass'] = penguins['BodyMass']/100

# The dataset is too small to be useful for deep learning
# So we'll oversample it to increase its size
for i in range(1,3):
    penguins = penguins.append(penguins)

# Display a random sample of 10 observations
sample = penguins.sample(10)
sample
```
See the sample:

![image](https://user-images.githubusercontent.com/71245576/115159426-0a4ca880-a061-11eb-8f99-b19c4bca6208.png)

The Sepcies column is what we want to predict, the label. There are three classes in the label, encoded as 0,1, or 2. Now show the actual species of these classes.

```python
penguin_classes = ['Adelie', 'Gentoo', 'Chinstrap']
print(sample.columns[0:5].values, 'SpeciesName')
for index, row in penguins.sample(10).iterrows():
    print('[',row[0], row[1], row[2],row[3], int(row[4]), ']',penguin_classes[int(row[-1])])
```
Gentoo, Adelie, and Chinstrap are three species in this label.

Now separate the features and the label, split the data into validation data set and training data set.

```python
from sklearn.model_selection import train_test_split

features = ['CulmenLength','CulmenDepth','FlipperLength','BodyMass']
label = 'Species'
   
# Split data 70%-30% into training set and test set
x_train, x_test, y_train, y_test = train_test_split(penguins[features].values,
                                                    penguins[label].values,
                                                    test_size=0.30,
                                                    random_state=0)

print ('Training Set: %d, Test Set: %d \n' % (len(x_train), len(x_test)))
print("Sample of features and labels:")

# Take a look at the first 25 training features and corresponding labels
for n in range(0,24):
    print(x_train[n], y_train[n], '(' + penguin_classes[y_train[n]] + ')')
```
The training set has 957 observations and the test set has 411 data set.

## 2. Preparation

This part is to install and import the PyTorch libraries and prepare the data for PyTorch.

First, install and import the PyTorch libraries we intend to use:
```python
!pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

It took a few minutes in Azure studio:

![image](https://user-images.githubusercontent.com/71245576/115159570-d1f99a00-a061-11eb-8d47-e865fb4a0118.png)

Now import PyTorch:

```python
import torch
import torch.nn as nn
import torch.utils.data as td

# Set random seed for reproducability
torch.manual_seed(0)

print("Libraries imported - ready to use PyTorch", torch.__version__)
```

PyTorch makes use of data loaders to load training and validatin data in batches. Because we have already loaded the data into numpy arrays, we need to wrap those in PyTorch dataset to convert those to PyTorch tensor objects and create loaders to read batches from those datasets.

Prepare the data for PyTorch:
```python
# Create a dataset and loader for the training data and labels
train_x = torch.Tensor(x_train).float()
train_y = torch.Tensor(y_train).long()
train_ds = td.TensorDataset(train_x,train_y)
train_loader = td.DataLoader(train_ds, batch_size=20,
    shuffle=False, num_workers=1)

# Create a dataset and loader for the test data and labels
test_x = torch.Tensor(x_test).float()
test_y = torch.Tensor(y_test).long()
test_ds = td.TensorDataset(test_x,test_y)
test_loader = td.DataLoader(test_ds, batch_size=20,
    shuffle=False, num_workers=1)
print('Ready to load data')
```

The data has been prepared.

## 3. Neural network modeling

We will create a network that consists of 3 fully connected layers(full propagating).

```python
# Number of hidden layer nodes
hl = 10

# Define the neural network
class PenguinNet(nn.Module):
    def __init__(self):
        super(PenguinNet, self).__init__()
        self.fc1 = nn.Linear(len(features), hl)
        self.fc2 = nn.Linear(hl, hl)
        self.fc3 = nn.Linear(hl, len(penguin_classes))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x),dim=1)
        return x

# Create a model instance from the network
model = PenguinNet()
print(model)
```

Now let's feed the training values forward through the network, use a loss function(backpropagating) to get the local optimum.

```python
def train(model, data_loader, optimizer):
    # Set the model to training mode
    model.train()
    train_loss = 0
    
    for batch, tensor in enumerate(data_loader):
        data, target = tensor
        #feedforward
        optimizer.zero_grad()
        out = model(data)
        loss = loss_criteria(out, target)
        train_loss += loss.item()

        # backpropagate
        loss.backward()
        optimizer.step()

    #Return average loss
    avg_loss = train_loss / (batch+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss
           
            
def test(model, data_loader):
    # Switch the model to evaluation mode (so we don't backpropagate)
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        batch_count = 0
        for batch, tensor in enumerate(data_loader):
            batch_count += 1
            data, target = tensor
            # Get the predictions
            out = model(data)

            # calculate the loss
            test_loss += loss_criteria(out, target).item()

            # Calculate the accuracy
            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target==predicted).item()
            
    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss/batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    
    # return average loss for the epoch
    return avg_loss

# Specify the loss criteria (CrossEntropyLoss for multi-class classification)
loss_criteria = nn.CrossEntropyLoss()

# Use an "Adam" optimizer to adjust weights
# (see https://pytorch.org/docs/stable/optim.html#algorithms for details of supported algorithms)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()

# We'll track metrics for each epoch in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 50 epochs
epochs = 50
for epoch in range(1, epochs + 1):

    # print the epoch number
    print('Epoch: {}'.format(epoch))
    
    # Feed training data into the model to optimize the weights
    train_loss = train(model, train_loader, optimizer)
    
    # Feed the test data into the model to check its performance
    test_loss = test(model, test_loader)
    
    # Log the metrics for this epoch
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)
```

We trained over 50 epochs, in each epoch the full set of training data is passed forward through the network. 

![image](https://user-images.githubusercontent.com/71245576/115159840-1b96b480-a063-11eb-9e66-9af009124bc0.png)

We can examine the loss metrics we recorded while training and validating the model. We are really looking for two things:

The loss should reduce with each epoch and the training loss and validation loss should follow a similar trend:

```python
%matplotlib inline
from matplotlib import pyplot as plt

plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
```

It performs well:

![image](https://user-images.githubusercontent.com/71245576/115159969-cc9d4f00-a063-11eb-9b72-004c212313e6.png)

We can view the learned weights and biases for each layer:

```python
for param_tensor in model.state_dict():
    print(param_tensor, "\n", model.state_dict()[param_tensor].numpy())
```
![image](https://user-images.githubusercontent.com/71245576/115160013-fb1b2a00-a063-11eb-8992-4d01fa24c7f9.png)

![image](https://user-images.githubusercontent.com/71245576/115160024-08381900-a064-11eb-8da7-843e850b1fac.png)

The raw accuracy reported from the validation data would seem to indicate that our model predicts pretty well; but it is typically useful to dig a litter deeper and compare the predictons for each possible class. A common way to visualize the performance of a classification model is to create a confusion matrix, as we have discussed previously.

```python
#Pytorch doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
from sklearn.metrics import confusion_matrix
import numpy as np

# Set the model to evaluate mode
model.eval()

# Get predictions for the test data
x = torch.Tensor(x_test).float()
_, predicted = torch.max(model(x).data, 1)

# Plot the confusion matrix
cm = confusion_matrix(y_test, predicted.numpy())
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(penguin_classes))
plt.xticks(tick_marks, penguin_classes, rotation=45)
plt.yticks(tick_marks, penguin_classes)
plt.xlabel("Actual Species")
plt.ylabel("Predicted Species")
plt.show()
```

The confusion matrix should show a strong diagonal line indicating that there are more correct than incorrect predictions for each class.

![image](https://user-images.githubusercontent.com/71245576/115160101-6cf37380-a064-11eb-956a-b6b08056ea53.png)

Now save the trained model:

```python
# Save the model weights
model_file = 'models/penguin_classifier.pt'
torch.save(model.state_dict(), model_file)
del model
print('model saved as', model_file)
```

Use the trained model to predict the label for new observation:

```python
# New penguin features
x_new = [[50.4,15.3,20,50]]
print ('New sample: {}'.format(x_new))

# Create a new model class and load weights
model = PenguinNet()
model.load_state_dict(torch.load(model_file))

# Set model to evaluation mode
model.eval()

# Get a prediction for the new data sample
x = torch.Tensor(x_new).float()
_, predicted = torch.max(model(x).data, 1)

print('Prediction:',penguin_classes[predicted.item()])
```

The prediction shows that the species is Gentoo.


## Reference

Train and evaluate deep learning models, retrieved from https://docs.microsoft.com/en-us/learn/modules/train-evaluate-deep-learn-models/



