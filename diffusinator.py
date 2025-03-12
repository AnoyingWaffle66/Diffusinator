import torch
import numpy as np
import model as m

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = m.NeuralNetwork().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.0005)

input = torch.rand(1, 4 * 16 * 16)
output = torch.rand(1, 4 * 16 * 16)
print(output)
images = []
for epoch in range(20000):
    optimizer.zero_grad()
    outputs = model(input)
    loss = criterion(outputs, output)
    loss.backward()
    optimizer.step()
    
    if(epoch + 1) % 2000 == 0:
        thingy = outputs.detach()
        images.append(np.array(thingy))
        print(f'Epoch [{epoch+1}/{10000}], Loss: {loss.item():.4f}')
    if epoch == 19999:
        print(outputs)
print(images)
print("Training complete.")
