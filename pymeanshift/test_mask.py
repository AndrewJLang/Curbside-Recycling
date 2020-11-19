from PIL import Image
import os

im = Image.open(os.path.join(os.path.dirname(__file__), "test_image.jpg"))
pix = im.load()

for x in range(im.size[0]):
    for y in range(im.size[1]):
        print(pix[x,y])