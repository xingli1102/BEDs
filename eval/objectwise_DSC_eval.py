import argparse
import os
import sys
import cv2
import numpy as np

from utils import *

'''
python objectwise_DSC_eval.py --ref-dir ../datasets/Test/Test_GT/ --input-dir ../experiments/fusing_results/EXPERIMENT_DIR/ --output-dir ../experiments/objectwise_F1/
'''

def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate objectwise DSC between two sets of mask')
	parser.add_argument(
		'--ref-dir',
		dest='ref_dir',
		help='directory of reference images',
		default='/tmp/infer_simple',
		type=str
	)
	parser.add_argument(
		'--input-dir',
		dest='input_dir',
		help='directory of input images',
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
		'--output-dir',
		dest='output_dir',
		help='directory of output images',
		default='/tmp/infer_simple',
		type=str
	)
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	return parser.parse_args()
	
def main(args):
	
	mkdir_if_nonexist(args.output_dir)
	output_path = os.path.join(args.output_dir, 'F1_sum.csv')
	csv_file = open(output_path, 'w')
	im_file_list = list_files(args.ref_dir, args.image_ext)
	true_positive_list = []
	false_positive_list = []
	false_negative_list = []
	for i, im_file in enumerate(im_file_list):
		ffname = os.path.splitext(os.path.basename(im_file))[0]
		# Read reference image and evaluate image
		img_ref_path = os.path.join(args.ref_dir, im_file)
		img_eval_path = os.path.join(args.input_dir, im_file)
		img_ref_bgr = cv2.imread(img_ref_path, -1)
		img_eval_bgr = cv2.imread(img_eval_path, -1)
		# Apply same watershed to both images
		img_ref_watershed, object_ref_list, cnt_center_ref_list = watershed_seg(img_ref_bgr)
		print("GT: " + str(len(object_ref_list)))
		img_eval_watershed, object_eval_list, cnt_center_eval_list = watershed_seg(img_eval_bgr)
		print("EVAL: " + str(len(object_eval_list)))
		ref_pair_list, eval_pair_list = get_obj_pair(object_ref_list, object_eval_list, cnt_center_ref_list, cnt_center_eval_list)
		print(len(ref_pair_list))
		print(len(eval_pair_list))
		true_positive, false_positive, false_negative = evaluation_objectwise(ref_pair_list, eval_pair_list)
		true_positive_list.append(true_positive)
		false_positive_list.append(false_positive)
		false_negative_list.append(false_negative)
		msg = ffname + " " + str(true_positive) + " " + str(false_positive) + " " + str(false_negative) + '\n'
		csv_file.write(msg)
		img_compare = np.concatenate([img_eval_watershed, img_ref_watershed], axis=1)
		img_out_path = os.path.join(args.output_dir, im_file)
		cv2.imwrite(img_out_path, img_compare)
	tp = np.sum(true_positive_list)
	fp = np.sum(false_positive_list)
	fn = np.sum(false_negative_list)
	precision = float(tp) / (float(tp) + float(fp))
	recall = float(tp) / (float(tp) + float(fn))
	f1 = 2 * precision * recall / (precision+recall)
	msg = "Precision" + " " + str(precision) + " " + "Recell" + " " + str(recall) + " " + "F1" + " " + str(f1) + '\n'
	print(msg)
	csv_file.write(msg)
	csv_file.close()
	return 0
	
if __name__ == '__main__':
	args = parse_args()
	main(args)
