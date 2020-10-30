import argparse
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import imutils
import time

from utils import *

'''
Running:
python benchmark_eval.py --nuclei-model ../models/benchmark/frozen_model.pb --output-dir ../experiments/benchmark/ ../datasets/Test/Test_pairs/0/
python benchmark_eval.py --nuclei-model ../models/deep_forest/random1/frozen_model.pb --output-dir ../experiments/random1/ ../datasets/Test/Test_pairs/0/
'''

def parse_args():
	parser = argparse.ArgumentParser(description='End-to-end inference')
	parser.add_argument(
		'--nuclei-model',
		dest='nuclei_model_file',
		help='Frozen TensorFlow model file.',
		default='../models/benchmark/frozen_model.pb',
		type=str
	)
	parser.add_argument(
		'--output-dir',
		dest='output_dir',
		help='directory for visualization files (default: /tmp/infer_simple)',
		default='../experiments/benchmark/',
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
		'im_or_folder',
		help='image or folder of images',
		default='../datasets/Test/Test_normed/0/',
		type=str
	)
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	return parser.parse_args()
	
def main(args):
	
	start = time.time()
	
	# Initialization
	mkdir_if_nonexist(args.output_dir)
	im_file_list = list_files(args.im_or_folder, args.image_ext)
	
	# Do Inference
	avg_DSC, std_DSC, max_DSC, min_DSC = infer_a_model(args.nuclei_model_file, im_file_list, args.im_or_folder, args.output_dir)
	
	time_used = time.time() - start
	print(time_used)
	
	return 0


if __name__ == '__main__':
	args = parse_args()
	main(args)
