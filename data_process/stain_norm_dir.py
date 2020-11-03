import argparse
import os
import sys
import spams
import staintools
import cv2

'''
Running:
python stain_norm_dir.py --input_dir ../datasets/Test/Images/ --target_dir stain_template/ --output_dir ../datasets/Test/Test_pairs_final/ --ext tif
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Stain Normalization')
    parser.add_argument(
        '--input_dir',
        dest='input_dir',
        help='Input Directory for Images',
        default='./HE',
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
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def mkdir_if_nonexist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))

def main(args):
    target_list = list_files(args.target_dir, 'png')
    for j, target_fname in enumerate(target_list):
        target_ffname = os.path.splitext(os.path.basename(target_fname))[0]
        print("Target: " + target_ffname)
        target_filepath = os.path.join(args.target_dir, target_fname)
        target = staintools.read_image(target_filepath)
        target = staintools.LuminosityStandardizer.standardize(target)
        normalizer = staintools.StainNormalizer(method='vahadane')
        normalizer.fit(target)
        output_dir_target = os.path.join(args.output_dir, target_ffname)
        mkdir_if_nonexist(output_dir_target)
        image_list = list_files(args.input_dir, args.ext)
        for i, image_fname in enumerate(image_list):
            image_ffname = os.path.splitext(os.path.basename(image_fname))[0]
            print("Normalizing: " + image_fname)
            image_filepath = os.path.join(args.input_dir, image_fname)
            to_transform = staintools.read_image(image_filepath)
            to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
            transformed = normalizer.transform(to_transform)
            transformed_BGR = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir_target, image_ffname+'.png'), transformed_BGR)
    
    return 0

if __name__ == '__main__':
    args = parse_args()
    main(args)
