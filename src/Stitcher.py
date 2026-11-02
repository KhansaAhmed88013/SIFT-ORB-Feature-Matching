'''
Stitching sample
================
 
Show how to use Stitcher API from python in a simple way to stitch panoramas
or scans.
'''

from __future__ import print_function

import numpy as np
import cv2 as cv

import argparse
import sys

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

parser = argparse.ArgumentParser(prog='stitching.py', description='Stiching sample.')

parser.add_argument('--mode',
                    type = int, choices = modes, 
                    default= cv.Stitcher_PANORAMA,
                    help='Determines configuration of stitcher. The default is `PANORAMA` (%d), '
                'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
                'for stitching materials under affine transformation, such as scans.' % modes)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'results' / 'stitched_result.jpg'

parser.add_argument('--output', default = str(DEFAULT_OUTPUT),
            help= 'Resulting image. the default is `result.jpg`.')
parser.add_argument('img', nargs='+', help = 'input images')

__doc__ += "\n" + parser.format_help()

def main():
    args = parser.parse_args()

    #read input images
    imgs = []
    for img_name in args.img:
        img = cv.imread(cv.samples.findFile(img_name))
        if img is None:
            print("cant read image" + img_name)
            sys.exit(-1)
        imgs.append(img)
    
    #![stitching] important part
    stitcher = cv.Stitcher.create(args.mode)
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print("can't stitch images, error code = %d" %status)
        sys.exit(-1)

    #![stitching]

    cv.imwrite(args.output, pano)
    print("stitching completed. %s saved!" % args.output)

    print('Done')

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()