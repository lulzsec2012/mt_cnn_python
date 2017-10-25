import argparse
import sys
import os
import cv2
#import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES','12')
from core.model import P_Net, R_Net, O_Net

from core.detector import Detector
from core.fcn_detector import FcnDetector
from core.MtcnnDetector import MtcnnDetector

def test_net( dataset_path, prefix, epoch,
              batch_size, test_mode="onet",
              thresh=[0.6, 0.6, 0.99], min_face_size=24,margin=44,
              stride=2, slide_window=False):

    detectors = [None, None, None]

    model_path=['%s-%s'%(x,y) for x,y in zip(prefix,epoch)]
    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0],model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    #load rnet model
    if test_mode in ["rnet", "onet"]:
        detectors[1]  = Detector(R_Net, 24, batch_size[1], model_path[1])
    # load onet model
    if test_mode == "onet":
        detectors[2] = Detector(O_Net, 48, batch_size[2], model_path[2])

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)
    ##load data
    input_dir0=dataset_path#/home/ssd/fb_data/jz_80val_67/'
    classes1 = os.listdir(input_dir0)

    for cls1 in classes1 :
      classes2_path = os.path.join(input_dir0, cls1 )
      try :
          classes2 = os.listdir(classes2_path)
      except Exception as e:
          print e
          continue

      img_list_tmp = []
      
      for cls2 in classes2 :
        classes3_path = os.path.join(classes2_path, cls2 )

        try :
            img = cv2.imread(classes3_path)
        except Exception as e:
            print e
            continue

        try :
            boxes, boxes_c = mtcnn_detector.detect_pnet(img)
        except Exception as e:
            print(classes3_path )
            continue
        if  boxes_c is  None:
            continue
        boxes, boxes_c = mtcnn_detector.detect_rnet(img, boxes_c)
        if  boxes_c is  None:
            continue
        boxes, boxes_c = mtcnn_detector.detect_onet(img, boxes_c)
        if  boxes_c is  None:
            continue

        if boxes_c is not None:
            box_count=0
            draw = img.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            for b in boxes_c:
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 1)
                cv2.putText(img, '%.3f'%b[4], (int(b[0]), int(b[1])), font, 0.4, (255, 255, 255), 1)
                box_count=box_count+1
            print 'boxs_count:',box_count

            cv2.imshow("detection result", img)
            cv2.waitKey(500) ## wait 500ms
       

def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='./jz_80val_0', type=str)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='onet', type=str)
    parser.add_argument('--detect_model', dest='detect_model', help='detect_model of model name', nargs="+",
                        default=['./model/wider_model/pnet', './model/wider_model/rnet', './model/wider_model/onet'], type=str)

    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[16, 16, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[1024, 128, 8], type=int)###default=[2048, 256, 16], type=int)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=0)#44)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+", default=[0.7, 0.8, 0.99], type=float)

    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=48, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    test_net(args.dataset_path, args.detect_model,
             args.epoch, args.batch_size, args.test_mode,
             args.thresh, args.min_face, args.stride,
             args.slide_window)

# python ./mtcnn_test.py

#CUDA_VISIBLE_DEVICES=0 python ./mtcnn_test.py
