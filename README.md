# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET


## Neural Network Model
Include the neural network model diagram.
<img width="1020" height="745" alt="image" src="https://github.com/user-attachments/assets/61b39b89-a2b6-4773-a0b7-4301bd63cc14" />


## DESIGN STEPS
### STEP 1: 
Load and Preprocess Data

### STEP 2: 
Get the shape of the first image in the training dataset


### STEP 3: 
Get the shape of the first image in the test dataset



### STEP 4: 
Train the model



### STEP 5: 
test the model



### STEP 6: 
predict on a single image and display the image




## PROGRAM

### Name:Santhosh G

### Register Number:21223240152

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name:Santhosh G')
        print('Register Number: 212223240152')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')



```

### OUTPUT

## Training Loss per Epoch
<img width="307" height="198" alt="image" src="https://github.com/user-attachments/assets/4ad3fab5-b588-4ef7-952a-19717bc4901c" />


## Confusion Matrix
<img width="755" height="759" alt="image" src="https://github.com/user-attachments/assets/e9dce19f-ee09-40bc-b225-786d65e5f449" />



## Classification Report
<img width="545" height="399" alt="image" src="https://github.com/user-attachments/assets/b7331ed9-a36d-4ce3-b62f-de04de1cd7b9" />


### New Sample Data Prediction
<img width="558" height="567" alt="image" src="https://github.com/user-attachments/assets/4b5612a2-0154-44ed-847b-4aeed3d4269b" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
