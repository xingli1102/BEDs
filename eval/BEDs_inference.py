import argparse
import os
import sys
import cv2
import numpy as np
import imutils
import subprocess
from progress.bar import Bar
import time

from utils import *

'''
Running:
python BEDs_inference.py --model-dir ../models/deep_forest/ --output-dir ../experiments/BEDs_inference_results/ ../datasets/Test/Test_pairs/
'''

def parse_args():
	parser = argparse.ArgumentParser(description='End-to-end inference')
	parser.add_argument(
		'--model-dir',
		dest='model_dir',
		help='directory stores random forrest model.',
		default='../models/deep_forest/',
		type=str
	)
	parser.add_argument(
		'--output-dir',
		dest='output_dir',
		help='directory for visualization files (default: /tmp/infer_simple)',
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
		'im_or_folder', help='image or folder of images', default=None
	)
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	return parser.parse_args()
	
def main(args):
	
	# Iterate over models and inference on image patches
	model_list = list_subfolder(args.model_dir)[0]
	model_list = sorted(model_list, key = lambda x: int(x[6:]))
	output_path = os.path.join(args.output_dir, 'performance_summary.csv')
	mkdir_if_nonexist(args.output_dir)
	summary_file = open(output_path, 'w')
	for n, ckpt_dir in enumerate(model_list):
		stain_list = list_subfolder(args.im_or_folder)[0]
		stain_list = sorted([int(x) for x in stain_list])
		
		for s in range(len(stain_list)):
			stain_folder = str(stain_list[s])
			stain_path = os.path.join(args.im_or_folder, stain_folder)
			# Get test image patches
			im_file_list = list_files(stain_path, args.image_ext)
			nuclei_model_file = os.path.join(args.model_dir, os.path.join(ckpt_dir, 'frozen_model.pb'))
			output_dir = os.path.join(os.path.join(args.output_dir, ckpt_dir), stain_folder)
			mkdir_if_nonexist(output_dir)
			print("Inference Model %s for Stain %s." % (ckpt_dir, stain_folder))
			dice_avg, dice_stdev, dice_max, dice_min = infer_a_model(nuclei_model_file, im_file_list, stain_path, output_dir)
			msg = ckpt_dir + " " + stain_folder + " " + str(dice_avg) + " " + str(dice_stdev) + " " + str(dice_max) + " " + str(dice_min) + '\n'
			print(msg)
			summary_file.write(msg)
	summary_file.close()
	
	return 0

if __name__ == '__main__':
	args = parse_args()
	main(args)
