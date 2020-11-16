import argparse
import os
import sys
import cv2
import numpy as np
import imutils
import subprocess
from progress.bar import Bar
import time
import glob

lib_path = os.getcwd()
sys.path.insert(0, os.path.join(lib_path, 'eval'))

from utils import *

def parse_args():
	parser = argparse.ArgumentParser(description='Stain Normalization')
	parser.add_argument(
		'--model_dir',
		dest='model_dir',
		help='directory stores random forrest model.',
		default='../models/deep_forest/',
		type=str
	)
	parser.add_argument(
		'--target_dir',
		dest='target_dir',
		help='Target stain transform Image',
		default='./Images',
		type=str
	)
	parser.add_argument(
		'--annot_dir',
		dest='annot_dir',
		help='Annotation directory for ground truth masks',
		default=None,
		type=str
	)
	parser.add_argument(
		'--output_dir',
		dest='output_dir',
		help='Output Directory for Normalized Images',
		default='./HE_Normed',
		type=str
	)
	parser.add_argument(
		'--ext',
		dest='ext',
		help='Image Extension Type',
		default='png',
	type=str
	)
	parser.add_argument(
		'display_partial_results',
		action="store_true",
		help="Save stain augmentation and single inference results."
	)
	parser.add_argument(
		'im_or_folder', help='image or folder of images, default dimension is 1000x1000', default=None
	)
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	return parser.parse_args()
	
def main(args):
	
	# Load images from a directory or a single file
	if os.path.isdir(args.im_or_folder):
		image_list = glob.iglob(args.im_or_folder + '/*.' + args.ext)
	else:
		image_list = [args.im_or_folder]
	ffdir = os.path.dirname(args.im_or_folder)
	
	for i, image_fname in enumerate(image_list):
		
		# Do stain augmentation for input image
		stainAug_dir = os.path.join(args.output_dir, 'Stain_Aug')
		mkdir_if_nonexist(stainAug_dir)
		print("Stain Augmentation ...")
		stain_aug_list = stainAug(image_fname, args.ext, args.target_dir, stainAug_dir, output_stain_augmentation=args.display_partial_results)
	
		# Do BEDs Inference for all sub models
		mask_dir = os.path.join(args.output_dir, 'Infer_Masks')
		mkdir_if_nonexist(mask_dir)
		print("Do inference ...")
		mask_list = BEDs_infer(stain_aug_list, image_fname, args.model_dir, mask_dir, output_stain_augmentation=args.display_partial_results)
		
		# Do Majority Vote for all masks
		print("Do Majority Vote ...")
		majorMask_bw, majorMask = majorVote_mask(mask_list)
		image_ffname = os.path.splitext(os.path.basename(image_fname))[0]
		majorMask_filepath = os.path.join(args.output_dir, image_ffname+'_pred.png')
		cv2.imwrite(majorMask_filepath, majorMask)
		
		# If annotation file was specified, do an evaluation
		if args.annot_dir != None:
			print("Do Evaluation ...")
			annot_filepath = os.path.join(args.annot_dir, image_ffname + '.png')
			annot = cv2.imread(annot_filepath, -1)
			annot_gray = cv2.cvtColor(annot, cv2.COLOR_BGR2GRAY)
			annot_bw = cv2.threshold(annot_gray, 127, 255, cv2.THRESH_BINARY)[1]
			majorDSC = dice(annot_bw, majorMask_bw)
			print(majorDSC)
		
	
if __name__ == '__main__':
	args = parse_args()
	main(args)
