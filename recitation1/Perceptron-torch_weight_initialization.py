import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputs = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],dtype=torch.float32).to(device)

majority = torch.tensor([0,0,0,1,0,1,1,1],dtype=torch.float32).to(device)
xor = torch.tensor([0,1,1,0,1,0,0,0],dtype=torch.float32).to(device)
onehotnot = torch.tensor([1,1,1,1,0,0,0,0],dtype=torch.float32).to(device)

class Perceptron(nn.Module):
    def __init__(self, input_size, activation_function):
        nn.Module.__init__(self)
        self.activation_function = activation_function
        self.l1 = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.l1(x)
        out = self.activation_function(out)
        return out
          


#activation_function = nn.ReLU()
#activation_function = nn.Sigmoid()
activation_function = nn.Tanh()


def weight_init_uniform_0_1(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.uniform_(layer.weight, a=0.0, b=1.0)
        torch.nn.init.constant_(layer.bias, 0.0)

def weight_init_uniform_neg1_1(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.uniform_(layer.weight, a=-1.0, b=1.0)
        torch.nn.init.constant_(layer.bias, 0.0)

def weight_init_zeros(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.constant_(layer.weight, 0.0)
        torch.nn.init.constant_(layer.bias, 0.0)

model = Perceptron(input_size=3, activation_function=activation_function).to(device)
model.apply(weight_init_zeros)
criterion = nn.MSELoss()
num_epochs = 100
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def non_batched_learning(labels):
    epoch_losses = []  #for plotting
    for epoch in range(num_epochs):
        i = 0
        total_loss = 0.0
        for input in inputs:
            output = model(input)
            label = labels[i].view(1)
            i += 1
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(inputs)
        epoch_losses.append(avg_loss)  # store average loss per epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return epoch_losses


def batched_learning(labels):

    split_labels = []
    for label in labels:
        split_labels.append([label])
    split_labels = torch.tensor(split_labels)

    for epoch in range(num_epochs*8):
        output = model(inputs)
        loss = criterion(output,split_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f'Epoch [{epoch + 1}/{num_epochs}]], Loss: {loss.item():.4f}')

def show():
    with torch.no_grad():
        for input in inputs:
            output = model(input)
            print(f"{input}->{output}")
            


losses = non_batched_learning(onehotnot)

#Plot the loss curve
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('onehotnot_Tanh Loss Curve with weight_init_zeros')
plt.grid(True)
plt.tight_layout()
plt.savefig("onehotnot_Tanh Loss Curve with weight_init_zeros.png")
#batched_learning(majority)

show()
print("\n Weights:", model.l1.weight)
print("Bias:", model.l1.bias)