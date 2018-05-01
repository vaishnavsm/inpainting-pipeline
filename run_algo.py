import numpy as np
import cv2
import sys
import os
import json
category_to_path = {"human":"celeba", "fallback":"imagenet", "place":"places2"}
if __name__ == '__main__':
    img_in = sys.argv[1]
    img_out = sys.argv[2]
    img_mask = ""
    if(len(sys.argv) > 3):
        img_mask = sys.argv[3]
    else:
        img_mask = img_out.split('.')[0]+"-mask.png"
    img_subdir = img_out.split('.')[0]+"-work"
    print("Making Directory...")
    os.system("mkdir %s"%img_subdir)
    print("Done, proceeding to Making Saliency Map and Threshold Map...")
    os.system("python split/saliency.py %s %s/saliency.png"%(img_in, img_subdir))
    print("Done. Proceeding to Making GrabCut Image and Map...")
    os.system("python split/grabcut.py %s %s/saliency.png %s/grabcut.png"%(img_in, img_subdir, img_subdir))
    print("Done. Proceeding to creating Telea Infill...")
    os.system("python tag/telea-infill.py %s/grabcut-image.png %s %s/telea-image.png"%(img_subdir, img_mask, img_subdir)) #Infills Completely white regions as these are probably holes
    print("Done. Proceeding to Classifying Foreground...")
    os.system("python tag/models/tutorials/image/imagenet/classify_image.py --image_file %s/telea-image.png --out_file %s/classifier-out.json"%(img_subdir, img_subdir))
    print("Done. Filling In Image Using Background Fill.")
    #os.system("python bg/inpainter %s %s -o %s/bg-fill.png"%(img_in, img_mask, img_subdir))
    print("Done. Finding Optimal Foreground Filler.")
    os.system("python tag/get-foreground-tagger.py %s/classifier-out.json %s/gan-to-use.json"%(img_subdir,img_subdir))
    print("Done. Running Optimal Foreground Filler.")
    try:
        file = open("%s/gan-to-use.json"%(img_subdir,), "r")
        jsonf = json.loads(file.read())
        path = category_to_path[jsonf['use']]
    except Exception as e:
        print("Error in getting saved category, using fallback")
        path = category_to_path['fallback']
    os.system("python fg/test.py --image %s --mask %s --output %s/gan-fill-0.png --checkpoint_dir fg/model_logs/%s"%(img_in, img_mask, img_subdir, path))
    print("Done. Stitching Images.")
    os.system("python stitch/stitch.py %s %s %s 1 %s"%(img_in, img_mask, img_subdir, img_out))
    print("Done! Inpaint Completed!")



