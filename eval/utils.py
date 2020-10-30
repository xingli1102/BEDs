import argparse
import os
import sys
import glob
import cv2
import math
import numpy as np
import tensorflow as tf
import imutils
from imutils import contours
from progress.bar import Bar
import time

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

# Global Var
infer_size = 256

# Create directory is not exists
def mkdir_if_nonexist(path):
	if not os.path.exists(path):
		os.makedirs(path)
		
# Find subfolder in a directory
def list_subfolder(directory):
	return [x[1] for x in os.walk(directory)]
	
# Find files in a directory with specified extension
def list_files(directory, extension):
	return (f for f in os.listdir(directory) if f.endswith('.' + extension))
	
# Random select a color from B,G,R
def random_color():
	rgb=[255,0,0]
	np.random.shuffle(rgb)
	return tuple(rgb)

# Compute l2 distance between two points
def compute_distance(A, B):
	x1 = A[0]
	y1 = A[1]
	x2 = B[0]
	y2 = B[1]
	dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
	return dist
	
# Load a (frozen) Tensorflow model into memory (.pb)
def load_graph(frozen_graph_filename):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph
	
def dice(true_mask, pred_mask, non_seg_score=1.0):
	"""
		Computes the Dice Coefficient.
		
		Args:
			true_mask: Array of arbitrary shape.
			pred_mask: Array with the same shape than true_mask.
			
		Returns:
			A scalar representing the Dice coefficient between the two segmentations.
	"""
	assert true_mask.shape == pred_mask.shape
	
	true_mask = np.asarray(true_mask).astype(np.bool)
	pred_mask = np.asarray(pred_mask).astype(np.bool)
	
	# If both segmentations are all zero, the dice will be 1.
	im_sum = true_mask.sum() + pred_mask.sum()
	if im_sum == 0:
		return non_seg_score
		
	# Compute Dice coefficient
	intersection = np.logical_and(true_mask, pred_mask)
	union = np.logical_or(true_mask, pred_mask)
	IoU = intersection.sum() / union.sum()
	return 2. * intersection.sum() / im_sum

def clean_mask_using_cnts(mask, cnts):
	'''
		Clean the predicted mask based on predicted contour information
		Args:
	'''
	clean_mask = np.zeros(mask.shape, np.uint8)
	clean_cnts = np.zeros(mask.shape, np.uint8)
	mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	mask_bw = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)[1]
	cnts_gray = cv2.cvtColor(cnts, cv2.COLOR_BGR2GRAY)
	cnts_bw = cv2.threshold(cnts_gray, 200, 255, cv2.THRESH_BINARY)[1]
	cnts = cv2.findContours(cnts_bw, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	try:
		cnts = contours.sort_contours(cnts)[0]
	except ValueError:
		cnts = cnts
	effective_contour = []
	for contour in cnts:
		contourArea = cv2.contourArea(contour)
		if contourArea > 20:
			effective_contour.append(contour)
	cv2.drawContours(clean_cnts, effective_contour, -1, (255,255,255), 2)
	clean_cnts_gray = cv2.cvtColor(clean_cnts, cv2.COLOR_BGR2GRAY)
	
	cnts = cv2.findContours(mask_bw, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	try:
		cnts = contours.sort_contours(cnts)[0]
	except ValueError:
		cnts = cnts
	effective_contour = []
	for contour in cnts:
		mask_temp_cnt = np.zeros(mask.shape, np.uint8)
		cv2.drawContours(mask_temp_cnt, [contour], -1, (255,255,255), -1)
		mask_temp_cnt_gray = cv2.cvtColor(mask_temp_cnt, cv2.COLOR_BGR2GRAY)
		mask_temp_cnt_bw = cv2.threshold(mask_temp_cnt_gray, 127, 255, cv2.THRESH_BINARY)[1]
		cnts_temp_gray = cv2.bitwise_and(clean_cnts_gray, clean_cnts_gray, mask=mask_temp_cnt_bw)
		cnts_sum = np.sum(cnts_temp_gray)
		if cnts_sum > 0:
			effective_contour.append(contour)
	cv2.drawContours(clean_mask, effective_contour, -1, (255,255,255), -1)
	return clean_mask, clean_cnts
	
def nuclei_detection_inference(img_src, nuclei_image_tensor, nuclei_output_tensor, nuclei_detection_session):
	'''
		Do nuclei segmentation inference on a testing image
		
		Args:
			img_src: intput image [1000,1000,3] 
			nuclei_image_tensor: input node on the graph
			nuclei_output_tensor: output node on the graph
			nuclei_detection_session: tensorflow session
			
		Returns:
			full_seg: 3 classes segmentation of the input image 
			clean_mask_gray: gray scale cleaned mask
			clean_cnts_gray: gray scale cleaned contour
			clean_mask_bw: binary cleaned mask
	'''
	# Initialization
	height, width = img_src.shape[:2]
	full_seg = np.zeros(img_src.shape, np.uint8)
	black_image = np.zeros((infer_size, infer_size, 3), np.uint8)
	# Create a frame for patch border clean
	frame_image_outter = np.zeros((infer_size, infer_size), np.uint8)
	frame_image_inner = 255*np.ones((infer_size-8, infer_size-8), np.uint8)
	frame_image_outter[4:infer_size-4, 4:infer_size-4]=frame_image_inner
	frame_image_bw = cv2.threshold(frame_image_outter, 0, 255, cv2.THRESH_BINARY)[1]
	# Inference input image by patch
	h_num = int(height / 250)
	w_num = int(width / 250)
	overlap_width = 20
	for i in range(h_num):
		for j in range(w_num):
			if i == h_num - 1 and i > 0:
				x0 = height - 250 - overlap_width
			else:
				x0 = i * infer_size
			x1 = min(height, (i+1) * 250 + overlap_width)
			if j == w_num - 1 and j > 0:
				y0 = width - 250 - overlap_width
			else:
				y0 = j * 250
			y1 = min(width, (j+1) * 250 + overlap_width)
			infer_patch = img_src[x0:x1, y0:y1].copy()
			infer_patch_height, infer_patch_width = infer_patch.shape[:2]
			infer_patch_resize = cv2.resize(infer_patch, (infer_size, infer_size), interpolation=cv2.INTER_AREA)
			infer_patch_rgb = cv2.cvtColor(infer_patch_resize, cv2.COLOR_BGR2RGB)
			combined_image = np.concatenate([infer_patch_rgb, black_image], axis=1)
			generated_image = nuclei_detection_session.run(nuclei_output_tensor, feed_dict={nuclei_image_tensor: combined_image})
			mask_patch = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
			# Clean false positive on predicted contour
			mask_patch_cnt = mask_patch[:,:,1].copy()
			mask_patch_cnt = cv2.bitwise_and(mask_patch_cnt, mask_patch_cnt, mask=frame_image_bw)
			mask_patch[:,:,1] = mask_patch_cnt.copy()
			# Resize and stitch the segmentation
			mask_patch_resize = cv2.resize(mask_patch, (infer_patch_width, infer_patch_height), interpolation=cv2.INTER_LINEAR)
			full_seg[x0:x1, y0:y1] = cv2.add(full_seg[x0:x1, y0:y1], mask_patch_resize)

	# Clean mask based on predicted contour information
	pred_mask = cv2.cvtColor(full_seg[:,:,2].copy(), cv2.COLOR_GRAY2BGR)
	pred_cnts = cv2.cvtColor(full_seg[:,:,1].copy(), cv2.COLOR_GRAY2BGR)
	clean_mask, clean_cnts = clean_mask_using_cnts(pred_mask, pred_cnts)
	clean_mask_gray = cv2.cvtColor(clean_mask, cv2.COLOR_BGR2GRAY)
	clean_cnts_gray = cv2.cvtColor(clean_cnts, cv2.COLOR_BGR2GRAY)
	clean_mask_bw = cv2.threshold(clean_mask_gray, 127, 255, cv2.THRESH_BINARY)[1]
	
	return full_seg, clean_mask_gray, clean_cnts_gray, clean_mask_bw
	
def infer_a_model(nuclei_model_file, im_file_list, im_or_folder, output_dir):

	# Loading tensorflow model for nuclei detection
	nuclei_detection_graph = load_graph(nuclei_model_file)
	nuclei_detection_session = tf.Session(graph=nuclei_detection_graph)
	print("[*] Nuclei Detection Model Loaded")
	# Get input and output tensor from graph for nuclei inference
	nuclei_image_tensor = nuclei_detection_graph.get_tensor_by_name('image_tensor:0')
	nuclei_output_tensor = nuclei_detection_graph.get_tensor_by_name('generate_output/output:0')
	
	# Initialization
	mask_dir = os.path.join(output_dir, 'mask')
	mkdir_if_nonexist(mask_dir)
	cnts_dir = os.path.join(output_dir, 'cnts')
	mkdir_if_nonexist(cnts_dir)
	DSC_list = []
	output_path = os.path.join(output_dir, 'DSC_summary.csv')
	csv_file = open(output_path, 'w')
	bar = Bar('Processing', max=14)
	
	for i, im_file in enumerate(im_file_list):
		# Stain Initialization
		ffname = os.path.splitext(os.path.basename(im_file))[0]
		# Get image and annotation
		img_combine = cv2.imread(os.path.join(im_or_folder, im_file), -1)
		img = img_combine[:,:1000,:].copy()
		annot = img_combine[:,1000:,:].copy()
		annot_mask = annot[:,:,2].copy()
		annot_cnts = annot[:,:,1].copy()
		annot_mask_bgr = cv2.cvtColor(annot_mask, cv2.COLOR_GRAY2BGR)
		annot_mask_bw = cv2.threshold(annot_mask, 127, 255, cv2.THRESH_BINARY)[1]
		# Inference segmentation on image
		mask_patch, pred_mask, pred_cnts_bw, pred_mask_bw = nuclei_detection_inference(img, nuclei_image_tensor, nuclei_output_tensor, nuclei_detection_session)
		pred_mask_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
		results = np.concatenate([pred_mask_bgr, np.concatenate([img, annot_mask_bgr], axis=1)], axis=1)
		output_fpath = os.path.join(output_dir, im_file)
		cv2.imwrite(output_fpath, results)
		mask_fpath = os.path.join(mask_dir, im_file)
		cv2.imwrite(mask_fpath, pred_mask_bw)
		cnts_fpath = os.path.join(cnts_dir, im_file)
		cv2.imwrite(cnts_fpath, pred_cnts_bw)
		DSC = dice(annot_mask_bw, pred_mask_bw)
		DSC_list.append(DSC)
		msg = ffname + " " + str(DSC) + '\n'
#		print(msg)
		csv_file.write(msg)
		bar.next()
		
		
	avg_DSC = np.mean(DSC_list)
	std_DSC = np.std(DSC_list)
	max_DSC = np.max(DSC_list)
	min_DSC = np.min(DSC_list)
	
	msg = "DICE_Average:" + " " + str(avg_DSC) + '\n'
	print(msg)
	csv_file.write(msg)
	msg = "DICE_Stdev:" + " " + str(std_DSC) + '\n'
	print(msg)
	csv_file.write(msg)
	msg = "DICE_Max:" + " " + str(max_DSC) + '\n'
	print(msg)
	csv_file.write(msg)
	msg = "DICE_Min:" + " " + str(min_DSC) + '\n'
	print(msg)
	csv_file.write(msg)
	
	nuclei_detection_session.close()
	csv_file.close()
	bar.finish()
	
	return avg_DSC, std_DSC, max_DSC, min_DSC
	
def majorVote_mask(mask_list):
	"""Do majority vote for an odd number of masks pixelwise."""
	mask_count = len(mask_list)
	sumMask = np.zeros(mask_list[0].shape)
	majorMask_bw = np.zeros(mask_list[0].shape, np.uint8)
	for i in range(mask_count):
		sumMask = sumMask + mask_list[i]
	intensity_thresh = int(255*(mask_count/2+1))
	sumMask[sumMask < intensity_thresh] = 0
	sumMask[sumMask > 0] = 255
	majorMask_bw = sumMask.copy()
	majorMask = cv2.cvtColor(majorMask_bw.astype(np.uint8), cv2.COLOR_GRAY2BGR)
	return majorMask_bw, majorMask
	
def watershed_seg(mask_bgr):
	mask_draw = mask_bgr.copy()
	mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	# Compute Euclidean distance from every binary pixel
	# to the nearest zero pixel then find peaks
	distance_map = ndimage.distance_transform_edt(thresh)
	local_max = peak_local_max(distance_map, indices=False, min_distance=12, labels=thresh)

	# Perform connected component analysis then apply Watershed
	markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
	labels = watershed(-distance_map, markers, mask=thresh)

	# Iterate through unique labels
	total_area = 0
	
	effective_cnts = []
	object_mask_list = []
	cnt_center_list = []
	for label in np.unique(labels):
		if label == 0:
		    continue
		object_mask = np.zeros(mask_bgr.shape, dtype="uint8")
		# Create a mask
		mask = np.zeros(mask_gray.shape, dtype="uint8")
		mask[labels == label] = 255
		# Find contours and determine contour area
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		c = max(cnts, key=cv2.contourArea)
		effective_cnts.append(c)
		cv2.drawContours(object_mask, [c], -1, (255,255,255), -1)
		# Draw contour on show image
		color = random_color()
		cv2.drawContours(mask_draw, [c], -1, color, -1)
		# Convert the object mask to binary and add it to the list
		object_mask_gray = cv2.cvtColor(object_mask, cv2.COLOR_BGR2GRAY)
		object_mask_bw = cv2.threshold(object_mask_gray, 127, 255, cv2.THRESH_BINARY)[1]
		object_mask_list.append(object_mask_bw)
		# Find cnt center
		M = cv2.moments(c)
		contourX = int(M["m10"]/max(0.1,M["m00"]))
		contourY = int(M["m01"]/max(0.1,M["m00"]))
		cnt_center_list.append([contourX, contourY])
	
	return mask_draw, object_mask_list, cnt_center_list
	
def dice_IoU(true_mask, pred_mask, non_seg_score=1.0):
	"""
		Computes the Dice Coefficient.
		
		Args:
			true_mask: Array of arbitrary shape.
			pred_mask: Array with the same shape than true_mask.
			
		Returns:
			A scalar representing the Dice coefficient between the two segmentations.
	"""
	assert true_mask.shape == pred_mask.shape
	
	true_mask = np.asarray(true_mask).astype(np.bool)
	pred_mask = np.asarray(pred_mask).astype(np.bool)
	
	# If both segmentations are all zero, the dice will be 1.
	im_sum = true_mask.sum() + pred_mask.sum()
	if im_sum == 0:
		return non_seg_score
		
	# Compute Dice coefficient
	intersection = np.logical_and(true_mask, pred_mask)
	union = np.logical_or(true_mask, pred_mask)
	IoU = intersection.sum() / union.sum()
	return 2. * intersection.sum() / im_sum, IoU
	
def evaluation_objectwise(ref_pair_list, eval_pair_list):
	true_positive = 0
	false_positive = 0
	false_negative = 0
	# Get True Positive and False Negative
	for i, ref_pair in enumerate(ref_pair_list):
		mask_ref = ref_pair[0]
		mask_eval = ref_pair[1]
		DSC, IoU = dice_IoU(mask_ref, mask_eval)
		if IoU > 0.5:
			true_positive += 1
		if IoU < 0.1:
			false_negative += 1
		print("Obj: " + str(i+1) + ", IoU: " + str(IoU) + ", TP: " + str(true_positive) + ", FP: " + str(false_positive) + ", FN: " + str(false_negative))
	print("TP: " + str(true_positive) + ", FP: " + str(false_positive) + ", FN: " + str(false_negative))
	# Get False Positive
	for i, eval_pair in enumerate(eval_pair_list):
		mask_ref = eval_pair[0]
		mask_eval = eval_pair[1]
		DSC, IoU = dice_IoU(mask_ref, mask_eval)
		if IoU < 0.1:
			false_positive += 1
		print("Obj: " + str(i+1) + ", IoU: " + str(IoU) + ", TP: " + str(true_positive) + ", FP: " + str(false_positive) + ", FN: " + str(false_negative))
	print("TP: " + str(true_positive) + ", FP: " + str(false_positive) + ", FN: " + str(false_negative))
	
	return true_positive, false_positive, false_negative

def get_obj_pair(object_ref_list, object_eval_list, cnt_center_ref_list, cnt_center_eval_list):
	# Get ref_pair_list
	ref_pair_list = []
	for i, cnt_center_ref in enumerate(cnt_center_ref_list):
		distance_list = []
		for j, cnt_center_eval in enumerate(cnt_center_eval_list):
			distance = compute_distance(cnt_center_eval, cnt_center_ref)
			distance_list.append(distance)
		min_id = np.argmin(distance_list)
		eval_select = object_eval_list[min_id]
		ref_pair_list.append([object_ref_list[i], eval_select])
		
	# Get eval_pari_list
	eval_pair_list = []
	for i, cnt_center_eval in enumerate(cnt_center_eval_list):
		distance_list = []
		for j, cnt_center_ref in enumerate(cnt_center_ref_list):
			distance = compute_distance(cnt_center_eval, cnt_center_ref)
			distance_list.append(distance)
		min_id = np.argmin(distance_list)
		ref_select = object_ref_list[min_id]
		eval_pair_list.append([ref_select, object_eval_list[i]])
		
	return ref_pair_list, eval_pair_list
