import argparse
import os
import sys
import glob
import cv2
import shutil
import numpy as np
import pandas
from pandas import DataFrame
import xml.etree.ElementTree as ET
import imutils
from imutils import contours
import random

'''
Train:
python data_process/prepare_datasets.py --annot-dir datasets/Train/Annotations/ --output-dir datasets/Train/deep_forest/ --stage train --subset-num 33 datasets/Train/Images/
Validation:
python data_process/prepare_datasets.py --annot-dir datasets/Val/Annotations/ --output-dir datasets/Val/Val/ --stage val datasets/Val/Images/
Test:
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotations/ --output-dir datasets/Test/Test_pairs/0/ --stage test datasets/Test/Images_stainNormed/0/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotations/ --output-dir datasets/Test/Test_pairs/1/ --stage test datasets/Test/Images_stainNormed/1/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotations/ --output-dir datasets/Test/Test_pairs/2/ --stage test datasets/Test/Images_stainNormed/2/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotations/ --output-dir datasets/Test/Test_pairs/3/ --stage test datasets/Test/Images_stainNormed/3/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotations/ --output-dir datasets/Test/Test_pairs/4/ --stage test datasets/Test/Images_stainNormed/4/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotations/ --output-dir datasets/Test/Test_pairs/5/ --stage test datasets/Test/Images_stainNormed/5/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotations/ --output-dir datasets/Test/Test_pairs/6/ --stage test datasets/Test/Images_stainNormed/6/
python data_process/prepare_datasets.py --annot-dir datasets/Test/Annotations/ --output-dir datasets/Test/Test_GT/ --stage test_gt datasets/Test/Images_stainNormed/0/
'''


def parse_args():
	parser = argparse.ArgumentParser(description='End-to-end inference')
	parser.add_argument(
		'--annot-dir',
		dest='annot_dir',
		help='directory for annotation files',
		default='/tmp/infer_simple',
		type=str
	)
	parser.add_argument(
		'--output-dir',
		dest='output_dir',
		help='directory for annotation files',
		default='/tmp/infer_simple',
		type=str
	)
	parser.add_argument(
		'--image-ext',
		dest='image_ext',
		help='image file name extension (default: jpg)',
		default='png',
		type=str
	)
	parser.add_argument(
		'--stage',
		dest='stage',
		required=True,
		choices=["train", "val", "test", "test_gt"]
	)
	parser.add_argument(
		'--subset-num',
		dest='subset_num',
		help='number of the subset for random forest',
		default=1,
		type=int
	)
	parser.add_argument(
		'im_or_folder', help='image or folder of images', default=None
	)
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	return parser.parse_args()

# Create directory is not exists
def mkdir_if_nonexist(path):
	if not os.path.exists(path):
		os.makedirs(path)

# List files in a directory with specified extension
def list_files(directory, extension):
	return (f for f in os.listdir(directory) if f.endswith('.' + extension))

# Process ground truth annotation from label map
def process_gt_labels(img, mask):
	'''
	Read segmentation ground truth from a 16-bit segmentation masks.
	Training dataset was made available from "Dataset of segmented nuclei in hematoxylin and eosin stained histopathology images of ten cancer types" (https://app.box.com/s/fz425ixs15kf56ghbnpxng1es6m7v2oh).
	Args:
		img:	original img in BGR format (color image).
		mask:	16-bit segmentation masks (label map).
		
	Output:
		mask_final:		 8-bit 3 classes segmentation, where blue is background, green is the boundary, red is the nucleus.
	'''
	# Load dataset from manual
	mask_8bit = np.zeros(mask.shape, np.uint8)
	mask_final = np.zeros(img.shape, np.uint8)
	# At least 1 nuclei in the training patch
	if np.max(mask) > 0:
		# Initialization
		gt_bg = 255 * np.ones(img.shape, np.uint8)
		gt_cnt = np.zeros(img.shape, np.uint8)
		gt_mask = np.zeros(img.shape, np.uint8)
		# Convert 16-bit to 8-bit
		ratio = 65535 / np.max(mask)
		mask_ratio = mask * ratio
		mask_8bit = (mask_ratio / 255).astype(np.uint8)
		# Go through all labels and draw the contours on final mask
		nuclei_index = np.sort(np.unique(mask_8bit))[1:]
		effective_contour = []
		for i in range(len(nuclei_index)):
			mask_temp = np.zeros(mask.shape).astype(np.uint8)
			mask_temp[mask_8bit==nuclei_index[i]] = 255
			mask_bw = cv2.threshold(mask_temp, 127, 255, cv2.THRESH_BINARY)[1]
			cnts = cv2.findContours(mask_bw, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			cnts = cnts[0] if imutils.is_cv2() else cnts[1]
			try:
				cnts = contours.sort_contours(cnts)[0]
			except ValueError:
				cnts = cnts
			for contour in cnts:
				effective_contour.append(contour)
		cv2.drawContours(gt_bg, effective_contour, -1, (0,0,0), -1)
		cv2.drawContours(gt_cnt, effective_contour, -1, (255,255,255), 2)
		cv2.drawContours(gt_mask, effective_contour, -1, (255,255,255), -1)
		gt_bg_1ch = cv2.cvtColor(gt_bg, cv2.COLOR_BGR2GRAY)
		gt_cnt_1ch = cv2.cvtColor(gt_cnt, cv2.COLOR_BGR2GRAY)
		gt_mask_1ch = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
		mask_final[:,:,0] = gt_bg_1ch	# Blue: Background
		mask_final[:,:,1] = gt_cnt_1ch	# Green: Boundary
		mask_final[:,:,2] = gt_mask_1ch	# Red: Nucleus
	# No nuclei in the training patch
	else:
		gt_bg = 255 * np.ones(img.shape, np.uint8)
		gt_bg_1ch = cv2.cvtColor(gt_bg, cv2.COLOR_BGR2GRAY)
		mask_final[:,:,0] = gt_bg_1ch
		
	return mask_final

# Process ground truth annotation from XML file
def process_gt_xml(img, annot_file):
	'''
	Read segmentation ground truth from an XML annotation file.
	Dataset can be downloaded from official MoNuSeg18 website.
	Args:
		img:		original img in BGR format (color image).
		annot_file:	xml file path of the ground truth segmentation.
		
	Output:
		gt_mask:	8-bit nucleus ground truth.
		gt_cnts:	8-bit contour ground truth.
		gt_final:	8-bit 3 classes segmentation, where blue is background, green is the boundary, red is the nucleus.
	'''
	# Initialization
	height, width = img.shape[:2]
	gt_mask = np.zeros(img.shape, np.uint8)
	gt_cnts = np.zeros(img.shape, np.uint8)
	gt_final = np.zeros(img.shape, np.uint8)
	gt_bg = 255 * np.ones(img.shape, np.uint8)
	img_draw = img.copy()
	# Read XML file and find segmenetation vertices
	tree = ET.parse(annot_file)
	annot = tree.find('Annotation')
	objs = annot.find('Regions')
	for obj in objs.findall('Region'):
		vertices = obj.find('Vertices')
		vertex_cnt = len(vertices.findall('Vertex'))
		cnts = np.zeros((vertex_cnt,1,2))
		# Initialize blank contour for to draw a single nuclei
		blackboard_cnt = np.zeros(img.shape, np.uint8)
		blackboard_mask = np.zeros(img.shape, np.uint8)
		count = 0
		# Go through all Regions and draw the contours on final mask
		for coord in vertices.findall('Vertex'):
			x = min(width, int(round(float(coord.attrib['X'])))) - 1
			y = min(height, int(round(float(coord.attrib['Y'])))) - 1
			cnts[count,0,0] = x
			cnts[count,0,1] = y
			count = count + 1
		color_temp = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
		cv2.drawContours(gt_mask, [cnts.astype(int)], -1, (255,255,255), -1)
		cv2.drawContours(gt_bg, [cnts.astype(int)], -1, (0,0,0), -1)
		cv2.drawContours(gt_cnts, [cnts.astype(int)], -1, (255,255,255), 2)
		cv2.drawContours(img_draw, [cnts.astype(int)], -1, (0,255,0), 1)
	gt_mask_1ch = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
	gt_cnts_1ch = cv2.cvtColor(gt_cnts, cv2.COLOR_BGR2GRAY)
	gt_bg_1ch = cv2.cvtColor(gt_bg, cv2.COLOR_BGR2GRAY)
	gt_final[:,:,0] = gt_bg_1ch		# Blue: Background
	gt_final[:,:,1] = gt_cnts_1ch	# Green: Boundary
	gt_final[:,:,2] = gt_mask_1ch	# Red: Nucleus
	return gt_mask, gt_cnts, gt_final
	
def split(data_dir, frac=0.66666667):
	
	files = glob.glob(os.path.join(data_dir, "*.png"))
	files.sort()
	assignments = []
	assignments.extend(["train"] * int(frac * len(files)))
	assignments.extend(["val"] * int(len(files) - len(assignments)))
	random.shuffle(assignments)
	
	train_dir = os.path.join(data_dir, "train")
	val_dir = os.path.join(data_dir, "val")
	mkdir_if_nonexist(train_dir)
	mkdir_if_nonexist(val_dir)
                
	for inpath, assignment in zip(files, assignments):
		outpath = os.path.join(data_dir, assignment, os.path.basename(inpath))
		os.rename(inpath, outpath)
	shutil.rmtree(val_dir)
		
	return 0
	
def main(args):
	# Make dataset for training data from manual label dataset
	if args.stage == "train":
		# set output directory for training subsets
		output_dir_list = []
		for num in range(args.subset_num):
			output_dir = os.path.join(args.output_dir, 'random' + str(num+1))
			output_dir_list.append(output_dir)
#			mkdir_if_nonexist(output_dir)
		# prepare benchmark training dataset
		benchmark_dir = os.path.join(args.output_dir, 'benchmark')
		mkdir_if_nonexist(benchmark_dir)
		im_file_list = list_files(args.im_or_folder, args.image_ext)
		im_file_list = sorted(im_file_list, key = lambda x: int(x[:-9]))
		for i, im_file in enumerate(im_file_list):
			ffname = os.path.splitext(os.path.basename(im_file))[0][:-5]
			img_ffname = ffname + '_crop.png'
			mask_ffname = ffname + '_labeled_mask_corrected.png'
			img_fpath = os.path.join(args.im_or_folder, img_ffname)
			mask_fpath = os.path.join(args.annot_dir, mask_ffname)
			img = cv2.imread(img_fpath, -1)
			mask = cv2.imread(mask_fpath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
			mask_final = process_gt_labels(img, mask)
			output_patch = np.concatenate([img, mask_final], axis=1)
			cv2.imwrite(os.path.join(benchmark_dir, mask_ffname), output_patch)
		# prepare subset training dataset
		for num, subset_dir in enumerate(output_dir_list):
			shutil.copytree(benchmark_dir, subset_dir)
			split(subset_dir)
		split(benchmark_dir, frac=1.0)
	# Mask dataset for validation and testing data from MoNuSeg18
	else:
		mkdir_if_nonexist(args.output_dir)
		im_file_list = list_files(args.im_or_folder, args.image_ext)
		for i, im_file in enumerate(im_file_list):
			ffname = os.path.splitext(os.path.basename(im_file))[0][:23]
			annot_filepath = os.path.join(args.annot_dir, ffname+'.xml')
			img_filepath = os.path.join(args.im_or_folder, im_file)
			# Loading Image
			img = cv2.imread(img_filepath, -1)
			img_copy = img.copy()
			height, width = img.shape[:2]
			gt_mask, gt_cnts, gt_final = process_gt_xml(img, annot_filepath)
			# Make dataset for validation - Divide large image in to 256x256 patches
			if args.stage == "val":
				h_num = int(height/250)
				w_num = int(width/250)
				for i in range(h_num):
					for j in range(w_num):
						x0 = i * 250
						x1 = min(height, (i+1) * 250)
						y0 = j * 250
						y1 = min(width, (j+1) * 250)
						img_patch = img_copy[x0:x1, y0:y1].copy()
						img_patch_256 = cv2.resize(img_patch, (256,256), interpolation=cv2.INTER_LINEAR)
						gt_cnts_patch = gt_final[x0:x1, y0:y1].copy()
						gt_cnts_patch_256 = cv2.resize(gt_cnts_patch, (256,256), interpolation=cv2.INTER_LINEAR)
						output_patch = np.concatenate([img_patch_256, gt_cnts_patch_256], axis=1)
						output_patch_path = os.path.join(args.output_dir, ffname + '_' + str(i) + str(j) + '.png')
						cv2.imwrite(output_patch_path, output_patch)
			# Make dataset for testing
			else:
				if args.stage == "test":
					output_pair = np.concatenate([img, gt_final], axis=1)
					output_patch_path = os.path.join(args.output_dir, ffname + '.png')
					cv2.imwrite(output_patch_path, output_pair)
				else:
					output_patch_path = os.path.join(args.output_dir, ffname + '.png')
					cv2.imwrite(output_patch_path, gt_mask)
	return 0

if __name__ == '__main__':
	args = parse_args()
	main(args)
