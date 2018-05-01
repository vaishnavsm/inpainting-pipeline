import sys
import cv2 as cv
import numpy as np

if __name__=="__main__":
    if(len(sys.argv)<6):
        print("Not Enough Arguments to stitch!")
        exit(-1)
    inp = sys.argv[1]
    mask = sys.argv[2]
    subdir = sys.argv[3]
    gan_no = int(sys.argv[4])
    output = sys.argv[5]

    img_inp = cv.imread(inp)
    img_mask = cv.imread(mask, 0)
    img_mask[img_mask>0] = 1 #Binarise Mask so we can take products
    img_gan = [cv.imread("%s/gan-fill-%s.png"%(subdir, x)) for x in range(gan_no)]
    img_gan = img_gan[0]
    img_gan_region = cv.imread("%s/grabcut.png"%(subdir,), 0)
    img_gan_region[img_gan_region>0] = 1
    img_bg = cv.imread("%s/bg-fill.png"%(subdir,))
    img_gan_region = np.multiply(img_gan_region, img_mask)
    img_out = img_inp.copy()
    for i in range(img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            if(img_gan_region[i][j]==1):
                img_out[i][j] = img_gan[i][j]
            elif(img_mask[i][j]==1):
                img_out[i][j] = img_bg[i][j]
            else:
                img_out[i][j] = img_inp[i][j]
    cv.imwrite(output, img_out)    
