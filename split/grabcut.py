import numpy as np
import cv2
import sys
def largest_contour_rect(saliency):
	image, contours, hierarchy = cv2.findContours(saliency * 1, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key = cv2.contourArea)
	return cv2.boundingRect(contours[-1])
def refine_saliency_with_grabcut(img, saliency):
	rect = largest_contour_rect(saliency)
	bgdmodel = np.zeros((1, 65),np.float64)
	fgdmodel = np.zeros((1, 65),np.float64)
	saliency[np.where(saliency > 0)] = cv2.GC_FGD
	mask = saliency
	mask, bgdModel, fgdModel = cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
	mask_bin = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	mask_img = np.where((mask==2)|(mask==0),0,255).astype('uint8')
	return mask_bin, mask_img
if __name__ == '__main__':
	img = cv2.imread(sys.argv[1], 1)
	img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
	mask = cv2.imread(sys.argv[2], 0)
	res_mask, res = refine_saliency_with_grabcut(img, mask)
	res_img = img*res_mask[:,:,np.newaxis]
	cv2.imwrite(sys.argv[3], res)
	cv2.imwrite(sys.argv[3].split(".")[0]+"-image.png", res_img)
