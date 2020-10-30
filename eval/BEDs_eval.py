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
BEDs 5 Model
python BEDs_eval.py --model-dir ../models/deep_forest/ --output-dir ../experiments/BEDs_inference_results/ --fusion-dir ../experiments/fusing_results/ --experiments BEDs_Model --model_num 5 ../datasets/Test/Test_pairs/0/
BEDs 33 Model
python BEDs_eval.py --model-dir ../models/deep_forest/ --output-dir ../experiments/BEDs_inference_results/ --fusion-dir ../experiments/fusing_results/ --experiments BEDs_Model --model_num 33 ../datasets/Test/Test_pairs/0/
BEDs 33 Model-Stain
python BEDs_eval.py --model-dir ../models/deep_forest/ --output-dir ../experiments/BEDs_inference_results/ --fusion-dir ../experiments/fusing_results/ --experiments BEDs_Model-Stain --model_num 33 ../datasets/Test/Test_pairs/0/
BEDs 33 Stain-Model
python BEDs_eval.py --model-dir ../models/deep_forest/ --output-dir ../experiments/BEDs_inference_results/ --fusion-dir ../experiments/fusing_results/ --experiments BEDs_Stain-Model --model_num 33 ../datasets/Test/Test_pairs/0/
BEDs 33 All
python BEDs_eval.py --model-dir ../models/deep_forest/ --output-dir ../experiments/BEDs_inference_results/ --fusion-dir ../experiments/fusing_results/ --experiments BEDs_All --model_num 33 ../datasets/Test/Test_pairs/0/
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
		'--fusion-dir',
		dest='fusion_dir',
		help='directory for visualization files (default: /tmp/infer_simple)',
		default='/tmp/infer_simple',
		type=str
	)
	parser.add_argument(
		'--experiments',
		dest='experiments',
		help='type of experiment for evaluation, choose from [BEDs_Model, BEDs_Stain-Model, BEDs_Model-Stain, BEDs_All]',
		choices=["BEDs_Model", "BEDs_Model-Stain", "BEDs_Stain-Model", "BEDs_All"],
		type=str
	)
	parser.add_argument(
		'--model_num',
		dest='model_num',
		help='number of models used for BEDs, choose from range [2, 33]',
		default=33,
		type=int
	)
	parser.add_argument(
		'--image-ext',
		dest='image_ext',
		help='image file name extension (default: jpg)',
		default='png',
		type=str
	)
	parser.add_argument(
		'im_or_folder', help='image or folder of images with ground truth', default=None
	)
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	return parser.parse_args()
	
def main(args):
	
	experiment = args.experiments
	model_num = args.model_num
	majority_dir = os.path.join(os.path.join(args.fusion_dir, experiment), experiment + str(model_num))
	mkdir_if_nonexist(majority_dir)
	output_path = os.path.join(majority_dir, 'test_eval.csv')
	summary_file = open(output_path, 'w')
	msg_header = "Image/Model" + " " + "DICE_Average" + " " + "DICE_Stdev" + " " + "DICE_Max" + " " + "DICE_Min" + '\n'
	summary_file.write(msg_header)
	
	testset_DSC_list = []
	
	for it in range(33):
		if model_num == 33:
			if it != 0:
				continue
		bar = Bar('Processing', max=14)
		# Get inference results and construct image list
		major_DSC_list = []
		im_file_list = list_files(args.im_or_folder, args.image_ext)
		model_used = []
		for p in range(model_num):
			model_used.append((it+p)%33)
		print("\nModel Applied:")
		print(model_used)
		for i, im_file in enumerate(im_file_list):
			start = time.time()
			# Load original image and annotation
			img_combine = cv2.imread(os.path.join(args.im_or_folder, im_file), -1)
			img = img_combine[:,:1000,:].copy()
			annot = img_combine[:,1000:,:].copy()
			annot_mask = annot[:,:,2].copy()
			annot_cnts = annot[:,:,1].copy()
			annot_mask_bgr = cv2.cvtColor(annot_mask, cv2.COLOR_GRAY2BGR)
			annot_mask_bw = cv2.threshold(annot_mask, 127, 255, cv2.THRESH_BINARY)[1]
			if experiment == "BEDs_Stain-Model":
				model_list = list_subfolder(args.output_dir)[0]
				model_list = sorted(model_list, key = lambda x: int(x[6:]))
				mask_png_list = []
				for n, model_id in enumerate(model_list):
					if n not in model_used:
						continue
					stain_list = list_subfolder(os.path.join(args.output_dir, model_id))[0]
					stain_list = sorted([int(x) for x in stain_list])
					mask_png_stain_list = []
					for s, stain_dir in enumerate(stain_list):
						stain_path = os.path.join(os.path.join(os.path.join(args.output_dir, model_id), str(stain_dir)), 'mask')
						png_filepath = os.path.join(stain_path, im_file)
						# Get binary mask image
						mask_gray = cv2.imread(png_filepath, -1)
						mask_bw = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)[1]
						mask_png_stain_list.append(mask_bw)
					print("Do Stain MV")
					majorMask_stain_bw, majorMask_stain = majorVote_mask(mask_png_stain_list)
					majorDSC_stain = dice(annot_mask_bw, majorMask_stain_bw)
					mask_png_list.append(majorMask_stain_bw)
				
				print("Do model MV")
				majorMask_bw, majorMask = majorVote_mask(mask_png_list)
				majorDSC = dice(annot_mask_bw, majorMask_bw)
				major_DSC_list.append(majorDSC)
				msg = im_file + " " + str(majorDSC) + '\n'
				summary_file.write(msg)
				print(msg)
				majorMask_filepath = os.path.join(majority_dir, im_file)
				cv2.imwrite(majorMask_filepath, majorMask)
				time_used = time.time() - start
				bar.next()
				print('\n')
						
			else:
				stain_list = list_subfolder(os.path.join(args.output_dir, 'random1'))[0]
				stain_list = sorted([int(x) for x in stain_list])
				mask_png_stain_list = []
				for s, stain_dir in enumerate(stain_list):
					if experiment == "BEDs_Model":
						if s != 0:
							continue
							
					mask_png_list = []
					model_list = list_subfolder(args.output_dir)[0]
					model_list = sorted(model_list, key = lambda x: int(x[6:]))
					for n, model_id in enumerate(model_list):
						if n not in model_used:
							continue
						model_mask_dir = os.path.join(os.path.join(os.path.join(args.output_dir, model_id), str(stain_dir)), 'mask')
						png_filepath = os.path.join(model_mask_dir, im_file)
	#					print(model_id)
						# Get binary mask image
						mask_gray = cv2.imread(png_filepath, -1)
						mask_bw = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)[1]
						if experiment == "BEDs_Model-Stain":
							mask_png_list.append(mask_bw)
						if experiment == "BEDs_All" or experiment == "BEDs_Model":
							mask_png_stain_list.append(mask_bw)
				
					if experiment == "BEDs_Model-Stain":
						print("Do model MV")
						# Do majority vote
						majorMask_bw, majorMask = majorVote_mask(mask_png_list)
						mask_png_stain_list.append(majorMask_bw)
						majorDSC = dice(annot_mask_bw, majorMask_bw)
			
				if experiment == "BEDs_Model-Stain":
					print("Do stain MV")
				if experiment == "BEDs_All" or experiment == "BEDs_Model":
					print("Do MV")
				majorMask_stain_bw, majorMask_stain = majorVote_mask(mask_png_stain_list)
				majorDSC_stain = dice(annot_mask_bw, majorMask_stain_bw)
				major_DSC_list.append(majorDSC_stain)
				msg = im_file + " " + str(majorDSC_stain) + '\n'
				summary_file.write(msg)
				print(msg)
				majorMask_filepath = os.path.join(majority_dir, im_file)
				cv2.imwrite(majorMask_filepath, majorMask_stain)
				time_used = time.time() - start
				bar.next()
				print('\n')
		
		avg_DSC = np.mean(major_DSC_list)
		std_DSC = np.std(major_DSC_list)
		max_DSC = np.max(major_DSC_list)
		min_DSC = np.min(major_DSC_list)
		if model_num != 33:
			testset_DSC_list.append(avg_DSC)
		msg = str(it) + " " + str(avg_DSC) + " " + str(std_DSC) + " " + str(max_DSC) + " " + str(min_DSC) + '\n'
		print(msg)
		summary_file.write(msg)
		bar.finish()
	
	if model_num != 33:
		avg_DSC = np.mean(testset_DSC_list)
		std_DSC = np.std(testset_DSC_list)
		max_DSC = np.max(testset_DSC_list)
		min_DSC = np.min(testset_DSC_list)
		msg = "Overall" + " " + str(avg_DSC) + " " + str(std_DSC) + " " + str(max_DSC) + " " + str(min_DSC) + '\n'
		print(msg)
		summary_file.write(msg)
	summary_file.close()

	return 0

if __name__ == '__main__':
	args = parse_args()
	main(args)
