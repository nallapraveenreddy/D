import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
x = torch.randn(100,10)
y = torch.randn(100,1)

#build the architecture
class SampleNet(nn.module):
    def__init__(self):
        super().__init__()
        self.fc1 =nn.Linear(10,100)
        self.fc2 =nn.Linear(100,50)
        self.fc3 =nn.Linear(50,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
model = SampleNet()
criterian = nn.MSEloss()
optimizer = optim.Adam(model.parameters(),lr = 0.0001)
losslist=[]
for epoch in range(500):
    predictions = model.forward(x)
    loss = criterian(predictions,y)
    loss.backward()
    optimizer.step()
    optim.zero_grad()
    print(f"loss-{epoch+1}:{loss.item}")
    losslist.append(loss)