import torch
import numpy as np
import model as m
import images as im

EPOCHS = 20000

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = m.NeuralNetwork().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.0015)

input = torch.rand(1, 4 * 32 * 32)
output = im.readImage("c:\\Users\\jarbon\\Downloads\\cirno.png")
print(output)
images = []
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(input)
    loss = criterion(outputs, output)
    loss.backward()
    optimizer.step()
    
    if(epoch + 1) % (EPOCHS // 10) == 0:
        thingy = outputs.detach()
        images.append(np.array(thingy))
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
    if epoch == EPOCHS - 1:
        print(outputs)

i = 1
for thing in images:
    thing = thing * 255
    thing = thing.astype(np.uint8)
    im.writeImage(f"{i}test.png", thing)
    i += 1

print(images)
print("Training complete.")
