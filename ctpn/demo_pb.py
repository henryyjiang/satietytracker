from __future__ import print_function

import glob
import os
import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from tensorflow.python.platform import gfile
import pytesseract
from PIL import Image
sys.path.append(os.getcwd())
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import _get_blobs
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(im_name):
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file('text.yml')

    # init session
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    sess = tf.compat.v1.Session(config=config)
    with gfile.FastGFile('data/ctpn.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.compat.v1.import_graph_def(graph_def, name='')
    sess.run(tf.compat.v1.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')

    img = cv2.imread(im_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    blobs, im_scales = _get_blobs(img, None)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
    rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

    scores = rois[:, 0]
    boxes = rois[:, 1:5] / im_scales[0]
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])

    base_name = im_name.split('\\')[-1]
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", base_name), img)
    #crop_image(base_name, 'data/results/res_{}.txt'.format(base_name.split('.')[0]))
    f = open(os.path.join("data/resulttext/{}.txt").format(base_name.split('.')[0]), "x")
    #f.write(pytesseract.image_to_string(os.path.join("data/results", base_name) , lang = 'eng')) # where img is
    f.write(pytesseract.image_to_string(img)) # where img is may not work
    f.close()
    # info = pytesseract.image_to_string('nf1.jpg' , lang = 'eng')

def crop_image(base_name, file_path):
    img = Image.open('{}'.format(base_name))
    count = 0
    # for loop
    with open(file_path, 'r') as file:
        for line in file:
            if len(line) != 1:
                # get each entry
                # Split the line by comma
                parts = line.strip().split(',')
                # Convert parts to integers
                params_int = [int(part) for part in parts]
                # crop
                # Call the function with the converted parameters
                cropped_image = img.crop(tuple(params_int))

                # save
                filename, extension = os.path.splitext(base_name)  # Splitting filename and extension
                cropped_filename = f"{filename}_cropped_{count}{extension}"  # Constructing new filename
                output_path = os.path.join("data/results/", cropped_filename)
                cropped_image.save(output_path)
                count += 1
# if '__name__' == '__main__':
#
#     if os.path.exists("data/results/"):
#         shutil.rmtree("data/results/")
#     os.makedirs("data/results/")
#
#     cfg_from_file('ctpn/text.yml')
#
#     # init session
#     config = tf.ConfigProto(allow_soft_placement=True)
#     sess = tf.Session(config=config)
#     with gfile.FastGFile('data/ctpn.pb', 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         sess.graph.as_default()
#         tf.import_graph_def(graph_def, name='')
#     sess.run(tf.global_variables_initializer())
#
#     input_img = sess.graph.get_tensor_by_name('Placeholder:0')
#     output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
#     output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
#
#     im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
#                glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))
#
#     for im_name in im_names:
#         print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#         print(('Demo for {:s}'.format(im_name)))
#         img = cv2.imread(im_name)
#         img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
#         blobs, im_scales = _get_blobs(img, None)
#         if cfg.TEST.HAS_RPN:
#             im_blob = blobs['data']
#             blobs['im_info'] = np.array(
#                 [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
#                 dtype=np.float32)
#         cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
#         rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)
#
#         scores = rois[:, 0]
#         boxes = rois[:, 1:5] / im_scales[0]
#         textdetector = TextDetector()
#         boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
#         draw_boxes(img, im_name, boxes, scale)
