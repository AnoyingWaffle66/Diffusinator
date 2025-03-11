import torch
import numpy as np
import model as m

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# print(model)

model = m.NeuralNetwork().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.005)

# input = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
# output = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
input = torch.tensor(torch.rand(1, 3 * 16))
output = torch.tensor(torch.rand(1, 3 * 16))
new_input = torch.tensor(torch.rand(1, 3 * 16))
# output = torch.tensor([[255/255, 192/255, 203/255]])
print(output)

for epoch in range(20000):
    optimizer.zero_grad()
    outputs = model([input, new_input])
    loss = criterion(outputs, output)
    loss.backward()
    optimizer.step()
    
    if(epoch + 1) % 2000 == 0:
        print(f'Epoch [{epoch+1}/{10000}], Loss: {loss.item():.4f}')
    if epoch == 19999:
        print(outputs)

outputs = model(new_input)
print(outputs)
# print(input)
print("Training complete.")



# print("Tensor-A:", A)

# X = A**3
# print("X:", X)

# X.backward(gradient=torch.tensor([255/255,192/255,203/255]))

# print("A.grad:", A.grad)