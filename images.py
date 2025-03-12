import torch
from PIL import Image
import numpy as np
import sys
import os

IMAGE_SIZE = 16, 16
FILE_NAME = os.path.basename(__file__)

def main():
    length = len(sys.argv)

    if (length == 1):
        print(f"{FILE_NAME} <input> [output]")
        return

    try:
        image = Image.open(sys.argv[1])

        if (image.size != IMAGE_SIZE):
            image = image.resize(IMAGE_SIZE, Image.LANCZOS)

        image = stripAlpha(image)
        array = np.array(image).flatten()
        tensor = torch.from_numpy(array)

        # TODO: Pass array to diffusinator.

        imageNew = Image.fromarray(np.reshape(tensor.numpy(), IMAGE_SIZE))

        output = sys.argv[2] if length > 2 else "new-" + os.path.basename(sys.argv[1])
        image.save(imageNew)

        print(f"Wrote to {output}")
    except IOError:
        print(f"{sys.argv[1]} cannot be read by PIL")
    except OSError:
        print("'if its an OS error then it's actually your fault' - Andrew Bell")

def stripAlpha(image):
    return image

if __name__ == "__main__":
    main()