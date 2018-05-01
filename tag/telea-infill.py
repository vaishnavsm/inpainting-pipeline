import numpy as np
import cv2
import sys

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], 1)
    if(img is None):
        raise Exception("Image Not Found")
        exit()
    if(len(img.shape)>2 and img.shape[2]>3):
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    greyscale[greyscale<245] = 0
    greyscale[greyscale>=245] = 1
    mask = cv2.imread(sys.argv[2], 0)
    mask[mask>0] = 1
    mask = np.multiply(greyscale, mask)
    mask[mask>0] = 255
    dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
    cv2.imwrite(sys.argv[3].split(".")[0]+'-telea-mask.png', mask)
    cv2.imwrite(sys.argv[3], dst)
