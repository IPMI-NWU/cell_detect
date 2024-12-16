import tensorflow as tf
import os
import numpy as np
import math
from .models import deeplab_model
model_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Model")
regionseg_weightspath =  os.path.join(model_root, "ki67_region_seg", "APN-Deeplab")
os.environ['CUDA_VISIBLE_DEVICES']='0'

num_class = 7
image_tensor = tf.placeholder(tf.float32, [None, None, 3])
inputs = deeplab_model.mean_image_subtraction(image_tensor)
inputs = tf.expand_dims(inputs, axis=0)
model = deeplab_model.deeplab_v3_plus_generator(num_classes=num_class,
                                                output_stride=16,
                                                base_architecture='resnet_v2_50',
                                                pre_trained_model=None,
                                                batch_norm_decay=None)
logits = model(inputs, is_training=False)
pred_classes = tf.nn.softmax(logits, axis=3)
pred_decoded_labels = tf.squeeze(pred_classes)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
model_file = tf.train.latest_checkpoint(regionseg_weightspath)
saver = tf.train.Saver()
saver.restore(sess, model_file)


def pad_img(img, pad_size=(512,512)):
    if img.shape[0:2] == pad_size:
        return img
    else:
        new_img = np.zeros((pad_size[0], pad_size[1], img.shape[2]))
        new_img[:img.shape[0], :img.shape[1],:] = img
        return new_img


def seg_turmor_region(img_np):
    patch_size = (512, 512)
    overlap = (64, 64)

    height, width = img_np.shape[0:2]
    crop_size_h, crop_size_w = patch_size
    overlap_h, overlap_w = overlap
    stride_h, stride_w = crop_size_h-overlap_h, crop_size_w-overlap_w
    # cut_border_thres = 1
    num_x, num_y = 1 + math.ceil((width-crop_size_w)/stride_w), 1 + math.ceil((height-crop_size_h)/stride_h)
    num_x, num_y = max(num_x, 1), max(num_y,1)

    seg_masks = np.zeros((height, width, 7))
    seg_masks_count = np.zeros((height, width, 7))

    for col_idx in range(num_x):
        for row_idx in range(num_y):
            # crop image
            ys, ye = row_idx*stride_h, row_idx*stride_h+crop_size_h
            xs, xe = col_idx*stride_w, col_idx*stride_w+crop_size_w
            cropImg = img_np[ys:ye, xs:xe, :]
            this_crop_h, this_crop_w = cropImg.shape[0:2]
            # pre-process
            padded_img = pad_img(cropImg, pad_size=patch_size)
            # generate mask
            output = sess.run(pred_decoded_labels, feed_dict={image_tensor: padded_img})
            seg_masks[ys:ye, xs:xe, :] += output[0:this_crop_h, 0:this_crop_w,:]
            seg_masks_count[ys:ye, xs:xe, :]+=1
    ave_seg_mask = seg_masks/seg_masks_count
    final_mask = ave_seg_mask[:,:,1] + ave_seg_mask[:,:,2]
    # final_mask = mask_post_process(final_mask)
    final_mask = final_mask>0.05
    return final_mask.astype(np.uint8)