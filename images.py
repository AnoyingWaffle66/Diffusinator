import torch
from PIL import Image
import numpy as np
import sys
import os

IMAGE_SIZE = 32, 32
FILE_NAME = os.path.basename(__file__)

def main():
    length = len(sys.argv)

    if (length == 1):
        print(f"{FILE_NAME} <input> [output]")
        return

    try:
        tensor = readImage(sys.argv[1])

        # TODO: Pass array to diffusinator.
        output = sys.argv[2] if length > 2 else "new-" + os.path.basename(sys.argv[1])

        writeImage(output, tensor)

        print(f"Wrote to {output}")
    except IOError:
        print(f"{sys.argv[1]} cannot be read by PIL")
    except OSError:
        print("'if its an OS error then it's actually your fault' - Andrew Bell")

def readImage(fileName, num):
    num = (num, num)
    image = Image.open(fileName).convert("RGBA")
    image.apply_transparency()
    if (image.size != num):
        image = image.resize(num, Image.LANCZOS)

    array = np.array(image).flatten()
    array = array.astype(np.float32) / 255.0
    return torch.from_numpy(array)

def writeImage(fileName, tensor, num):
    Image.fromarray(np.reshape(tensor, (num, num, 4))).save("images/" + fileName)

if __name__ == "__main__":
    main()