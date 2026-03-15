import torch 
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.model_selection import train_test_split 
import torch.nn as nn 
from torchinfo import summary
from utils import prepare_data
from time import time

class LogisticRegression(nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(LogisticRegression, self).__init__() 
        self.linear = nn.Linear(input_size, num_classes) 
  
    def forward(self, x): 
        out = self.linear(x) 
        out = nn.functional.softmax(out, dim=1) 
        return out 
    
def logreg_pytorch(batch_size=32, lr=0.1, epochs=1000):
    X, y = prepare_data()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

    train_dataset = TensorDataset(X_train, y_train) 
    val_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 

    model = LogisticRegression(input_size=1, num_classes=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) 

    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    start_time = time()

    for epoch in range(epochs): 
        for i, (inputs, labels) in enumerate(train_loader): 
            inputs = inputs.to(device) 
            labels = labels.to(device) 

            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
    
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
    
        # if (epoch+1)%100 == 0: 
        #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    end_time = time()

    print(f"Training time: {end_time-start_time}")

    for name, param in model.named_parameters():
        print(f"{name}: {param}")

    with torch.no_grad(): 
        correct = 0
        total = 0
        for inputs, labels in val_loader: 
            inputs = inputs.to(device) 
            labels = labels.to(device) 
    
            outputs = model(inputs) 
            _, predicted = torch.max(outputs.data, 1) 

            total += labels.size(0) 
            correct += (predicted == labels).sum().item() 
    
        print(f"Accuracy: {correct / total}")

if __name__ == "__main__":
    logreg_pytorch()