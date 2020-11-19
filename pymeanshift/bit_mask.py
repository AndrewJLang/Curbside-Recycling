from PIL import Image
import os
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

im = Image.open(os.path.join(os.path.dirname(__file__), "mask_adjusted_images/cardboard/image00018.jpg"))
pix = im.load()

def getBitArray(pixels, bitImage):
    arr = np.empty([bitImage.size[0], bitImage.size[1]], dtype=int)

    #Assert that the array is the same dimensions as image
    assert arr.shape[0] == bitImage.size[0]
    assert arr.shape[1] == bitImage.size[1]
    pixelSums = [0, 765]
    countBlack, countWhite = 0,0


    for x in range(bitImage.size[0]):
        for y in range(bitImage.size[1]):
            print(pixels[x,y])
            rgbVal = int(sum(list(pixels[x,y])))
            bitVal = pixelSums[min(range(len(pixelSums)), key = lambda i: abs(pixelSums[i] - rgbVal))]
            if bitVal == 0:
                arr[x][y] = 1
            else:
                arr[x][y] = 0

    return arr


getBitArray(pix, im)
