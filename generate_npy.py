#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use cuda and multiple GPUs.
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# python generate_tsv.py --net res101 --dataset vg --out test.csv --cuda


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import csv
import torch
import base64
from utils.timer import Timer
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from imageio import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
from tqdm import tqdm
import math
import spacy
nlp1 = spacy.load("en_core_web_lg")
nlp2 = spacy.load("en_vectors_web_lg")

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 36
MAX_BOXES = 36
# MIN_BOXES = 10
# MAX_BOXES = 100

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='vg', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="data/pre_trained_model")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images',
                        default="images")
    parser.add_argument('--classes_dir', dest='classes_dir',
                        help='directory to load object classes for classification',
                        default="data/genome/1600-400-20")
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default="object_representation_glove1.npy", type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default='karpathy_train', type=str)

    # if len(sys.argv) == 1:
    #     # parser.print_help()
    #     # sys.exit(1)

    args = parser.parse_args()

    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

#build [image_path, image_id] for dataset, and you can create your own
def load_image_ids(split_name):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []
    if split_name == 'coco_test2014':
      with open('/data/coco/annotations/image_info_test2014.json') as f:
        data = json.load(f)
        for item in data['images']:
          image_id = int(item['id'])
          filepath = os.path.join('/data/test2014/', item['file_name'])
          split.append((filepath,image_id))
    elif split_name == 'coco_test2015':
      with open('/data/coco/annotations/image_info_test2015.json') as f:
        data = json.load(f)
        for item in data['images']:
          image_id = int(item['id'])
          filepath = os.path.join('/data/test2015/', item['file_name'])
          split.append((filepath,image_id))
    elif split_name == 'genome':
      with open('/data/visualgenome/image_data.json') as f:
        for item in json.load(f):
          image_id = int(item['image_id'])
          filepath = os.path.join('/data/visualgenome/', item['url'].split('rak248/')[-1])
          split.append((filepath,image_id))
    else:
      print ('Unknown split')
    return split

def get_detections_from_im(fasterRCNN, classes, im_file, args, conf_thresh=0.2):
    """obtain the image_info for each image,
    im_file: the path of the image

    return: dict of {'image_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features'}
    boxes: the coordinate of each box
    """
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    #load images
    # im = cv2.imread(im_file)
    im_in = np.array(imread(im_file))
    if len(im_in.shape) == 2:
      im_in = im_in[:,:,np.newaxis]
      im_in = np.concatenate((im_in,im_in,im_in), axis=2)
    # rgb -> bgr
    im = im_in[:,:,::-1]

    vis = True

    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
    # pdb.set_trace()
    det_tic = time.time()

    # the region features[box_num * 2048] are required.
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, pooled_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, pool_feat = True)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
          if args.class_agnostic:
              if args.cuda > 0:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

              box_deltas = box_deltas.view(1, -1, 4)
          else:
              if args.cuda > 0:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
              box_deltas = box_deltas.view(1, -1, 4 * len(classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    det_toc = time.time()
    detect_time = det_toc - det_tic
    misc_tic = time.time()

    max_conf = torch.zeros((pred_boxes.shape[0]))
    if args.cuda > 0:
        max_conf = max_conf.cuda()

    if vis:
        im2show = np.copy(im)
    for j in xrange(1, len(classes)):
        inds = torch.nonzero(scores[:,j]>conf_thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
          cls_scores = scores[:,j][inds]
          _, order = torch.sort(cls_scores, 0, True)
          if args.class_agnostic:
            cls_boxes = pred_boxes[inds, :]
          else:
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

          cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
          # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
          cls_dets = cls_dets[order]
          # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
          keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
          cls_dets = cls_dets[keep.view(-1).long()]
          index = inds[order[keep]]
          max_conf[index] = torch.where(scores[index, j] > max_conf[index], scores[index, j], max_conf[index])
          if vis:
            im2show = vis_detections(im2show, classes[j], cls_dets.cpu().numpy(), 0.5)

    if args.cuda > 0:
        keep_boxes = torch.where(max_conf >= conf_thresh, max_conf, torch.tensor(0.0).cuda())
    else:
        keep_boxes = torch.where(max_conf >= conf_thresh, max_conf, torch.tensor(0.0))
    keep_boxes = torch.squeeze(torch.nonzero(keep_boxes), dim=-1)
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = torch.argsort(max_conf, descending = True)[:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = torch.argsort(max_conf, descending = True)[:MAX_BOXES]

    objects = torch.argmax(scores[keep_boxes][:,1:], dim=1)
    box_dets = np.zeros((len(keep_boxes), 4))
    boxes = pred_boxes[keep_boxes]
    name_list = []
    box_caption_feature = np.zeros((len(keep_boxes), 300))
    box_caption_mask = np.ones(len(keep_boxes))
    for i in range(len(keep_boxes)):
        kind = objects[i]+1
        bbox = boxes[i, kind * 4: (kind + 1) * 4]
        tmp_dets = np.array(bbox.cpu())
        if (tmp_dets[2]-tmp_dets[0]) * (tmp_dets[3]-tmp_dets[1]) <= 10:
             box_caption_mask[i] = 0
        class_name = classes[1:][objects[i]]
        box_dets[i] = tmp_dets
        name_list.append(class_name)
        doc = nlp1(class_name)
        token_vector = nlp2(doc[0].text).vector
        box_caption_feature[i,:] = token_vector

    return {
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        #'boxes': box_dets, # region shape 4 * 36, 4 is the xy positions
        #'features': (pooled_feat[keep_boxes].cpu()).detach().numpy(),
        'text': name_list,
        #'text_feature': box_caption_feature,
       # 'text_mask': box_caption_mask
    }
    # return {
    #     'image_h': np.size(im, 0),
    #     'image_w': np.size(im, 1),
    #     'num_boxes': len(keep_boxes),
    #     'boxes': box_dets, # region shape 4 * 36, 4 is the xy positions
    #     'features': (pooled_feat[keep_boxes].cpu()).detach().numpy()
    # }

def load_model(args):
    # set cfg according to the dataset used to train the pre-trained model
    if args.dataset == "pascal_voc":
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    if args.cfg_file is not None:
      cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
      cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # Load classes
    classes = ['__background__']
    with open(os.path.join(args.classes_dir, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())

    if not os.path.exists(args.load_dir):
        raise Exception('There is no input directory for loading network from ' + args.load_dir)
    load_name = os.path.join(args.load_dir, 'faster_rcnn_{}_{}.pth'.format(args.net, args.dataset))

    # initilize the network here. the network used to train the pre-trained model
    if args.net == 'vgg16':
      fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
      fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
      fasterRCNN = resnet(classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
      fasterRCNN = resnet(classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
      print("network is not defined")
      pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
      checkpoint = torch.load(load_name)
    else:
      checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    print("load model %s" % (load_name))

    return classes, fasterRCNN


def generate_npy(image_ids, classes, faster_rcnn):
    generated_data = {}
    
    for im_file, heading in image_ids:
        generated_data[heading] = get_detections_from_im(fasterRCNN, classes, im_file, args)
    return generated_data

def combine_npy(relative_generate_path):
    """
    generate each states npy seperately, and then combine them. You could generate directly.
    """
    feature_of_all_states = {}
    generated_files_list = os.listdir(relative_generate_path)
    for generated_file in tqdm(generated_files_list):
        each_state_dict = np.load(os.path.join(relative_generate_path, generated_file)).item()
        feature_of_all_states[generated_file[:-4]] = each_state_dict
        #feature_of_all_states.update(each_state_dict.item())

    print(len(feature_of_all_states.keys()))
    np.save(args.outfile, feature_of_all_states)

def get_heading_degree():
        new_headings = []
        for i in range(0, 360, 30):
            current_radians = i*math.pi/180
            new_headings.append(str(current_radians))
        return new_headings

if __name__ == '__main__':
    """
    data structure:
    {
        "10c252c90fa24ef3b698c6f54d984c5c": {0.03:{"image_w":360, "image_h":90, 'feature':''},
                                             0.3: {"image_w":360, "image_h":90, 'feature':''}},
        "0e92a69a50414253a23043758f111cec":	{0.05:{"image_w":360, "image_h":90, 'feature':''},
									        0.5:{"image_w":360, "image_h":90, 'feature':''}}
    }
    """
    args = parse_args()
    classes, fasterRCNN = load_model(args)
    
    relative_path = "/egr/research-hlr/joslin/Matterdata/v1/scans/generated_images/"
    #relative_path = "pre-trained"
    all_scenery = os.listdir(relative_path)
    
    all_scenery = ['rqfALeAoiTq']
    for scen_id, each_scenery in enumerate(all_scenery):
        feature_of_all_states = {}
        output_file_list = os.listdir("pre-trained/")
        if each_scenery in output_file_list:
            continue
        state_list = os.listdir(relative_path + each_scenery)
        for state_id in tqdm(state_list):
            each_state = {}
            state_path = relative_path + each_scenery + "/" + state_id
            image_ids = []
            heading_list = os.listdir(state_path)
            standard_headings = get_heading_degree()
            new_heading_list = []
            for each_heading in heading_list:
                each_heading = 'e9fa9291c1e140aabfe4bba03e64a9a1_0.0'
                tmp_heading = each_heading.split('_')
                if tmp_heading[-1][:-4] in standard_headings:
                    new_heading_list.append(each_heading)
            
            for each_image in new_heading_list:
                temp = []
                temp.append(os.path.join(state_path,each_image))
                each_image = each_image.split('_')
                heading = each_image[1][:-4]
                temp.append(float(heading))
                image_ids.append(temp)
            
            feature_of_all_states[state_id] = generate_npy(image_ids, classes, fasterRCNN)

        np.save("pre-trained/"+each_scenery+".npy", feature_of_all_states)
        print('finish scenery '+ str(scen_id) + " " + each_scenery)

    #combine_npy(relative_generate_path = "/egr/research-hlr/joslin/Matterdata/v1/scans/object_representation/pre-trained")