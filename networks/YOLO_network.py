import os
import numpy as np
import cv2
import tensorflow as tf
import time
from utils import ROLO_utils as util

class YOLO_TF:
    disp_console = True
    weights_file = 'weights/YOLO_small.ckpt'
    alpha = 0.1
    threshold = 0.08
    iou_threshold = 0.5
    classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    w_img, h_img = [352, 240]
    num_feat = 4096
    num_predict = 6 # final output of LSTM 6 loc parameters

    def __init__(self,argvs = []):
        self.build_networks()

    def build_networks(self):
        if self.disp_console : print "Building YOLO_small graph..."

        graph = tf.Graph()
        with graph.as_default():
            self.x = tf.placeholder('float32',[None,448,448,3])
            self.conv_1 = self.conv_layer(1,self.x,64,7,2)
            self.pool_2 = self.pooling_layer(2,self.conv_1,2,2)
            self.conv_3 = self.conv_layer(3,self.pool_2,192,3,1)
            self.pool_4 = self.pooling_layer(4,self.conv_3,2,2)
            self.conv_5 = self.conv_layer(5,self.pool_4,128,1,1)
            self.conv_6 = self.conv_layer(6,self.conv_5,256,3,1)
            self.conv_7 = self.conv_layer(7,self.conv_6,256,1,1)
            self.conv_8 = self.conv_layer(8,self.conv_7,512,3,1)
            self.pool_9 = self.pooling_layer(9,self.conv_8,2,2)
            self.conv_10 = self.conv_layer(10,self.pool_9,256,1,1)
            self.conv_11 = self.conv_layer(11,self.conv_10,512,3,1)
            self.conv_12 = self.conv_layer(12,self.conv_11,256,1,1)
            self.conv_13 = self.conv_layer(13,self.conv_12,512,3,1)
            self.conv_14 = self.conv_layer(14,self.conv_13,256,1,1)
            self.conv_15 = self.conv_layer(15,self.conv_14,512,3,1)
            self.conv_16 = self.conv_layer(16,self.conv_15,256,1,1)
            self.conv_17 = self.conv_layer(17,self.conv_16,512,3,1)
            self.conv_18 = self.conv_layer(18,self.conv_17,512,1,1)
            self.conv_19 = self.conv_layer(19,self.conv_18,1024,3,1)
            self.pool_20 = self.pooling_layer(20,self.conv_19,2,2)
            self.conv_21 = self.conv_layer(21,self.pool_20,512,1,1)
            self.conv_22 = self.conv_layer(22,self.conv_21,1024,3,1)
            self.conv_23 = self.conv_layer(23,self.conv_22,512,1,1)
            self.conv_24 = self.conv_layer(24,self.conv_23,1024,3,1)
            self.conv_25 = self.conv_layer(25,self.conv_24,1024,3,1)
            self.conv_26 = self.conv_layer(26,self.conv_25,1024,3,2)
            self.conv_27 = self.conv_layer(27,self.conv_26,1024,3,1)
            self.conv_28 = self.conv_layer(28,self.conv_27,1024,3,1)
            self.fc_29 = self.fc_layer(29,self.conv_28,512,flat=True,linear=False)
            self.fc_30 = self.fc_layer(30,self.fc_29,4096,flat=False,linear=False)
            # skip dropout_31
            self.fc_32 = self.fc_layer(32,self.fc_30,1470,flat=False,linear=True)
            self.sess = tf.Session(graph = graph)
            # self.sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            saver.restore(self.sess,self.weights_file)

        if self.disp_console : print "Loading complete!" + '\n'

    def conv_layer(self,idx,inputs,filters,size,stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size//2
        pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        inputs_pad = tf.pad(inputs,pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')
        conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')
        if self.disp_console : print '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels))
        return tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

    def pooling_layer(self,idx,inputs,size,stride):
        if self.disp_console : print '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride)
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

    def fc_layer(self,idx,inputs,hiddens,flat = False,linear = False):
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1]*input_shape[2]*input_shape[3]
            inputs_transposed = tf.transpose(inputs,(0,3,1,2))
            inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
        if self.disp_console : print '    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx,hiddens,int(dim),int(flat),1-int(linear))
        if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
        ip = tf.add(tf.matmul(inputs_processed,weight),biases)
        return tf.maximum(self.alpha*ip,ip,name=str(idx)+'_fc')

    def interpret_output(self,output):
        probs = np.zeros((7,7,2,20))
        class_probs = np.reshape(output[0:980],(7,7,20))
        scales = np.reshape(output[980:1078],(7,7,2))
        boxes = np.reshape(output[1078:],(7,7,2,4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])

        boxes[:,:,:,0] *= self.w_img
        boxes[:,:,:,1] *= self.h_img
        boxes[:,:,:,2] *= self.w_img
        boxes[:,:,:,3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

        filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0 : continue
            for j in range(i+1,len(boxes_filtered)):
                if self.iou(boxes_filtered[i],boxes_filtered[j]) > self.iou_threshold :
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return result

    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

    def debug_location(self, img, location):
        img_cp = img.copy()
        x = int(location[1])
        y = int(location[2])
        w = int(location[3])//2
        h = int(location[4])//2
        cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
        cv2.putText(img_cp, str(location[0]) + ' : %.2f' % location[5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        cv2.imshow('YOLO_small detection',img_cp)
        cv2.waitKey(0)

    def iou(self,box1,box2):
            tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
            lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
            if tb < 0 or lr < 0 : intersection = 0
            else : intersection =  tb*lr
            return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

    def find_best_location(self, locations, gt_location):
        # locations (class, x, y, w, h, prob); (x, y) is the middle pt of the rect
        # gt_location (x1, y1, w, h)
        x1 = gt_location[0]
        y1 = gt_location[1]
        w = gt_location[2]
        h = gt_location[3]
        gt_location_revised= [x1 + w/2, y1 + h/2, w, h]

        max_ious= 0
        for id, location in enumerate(locations):
                location_revised = location[1:5]
                ious = self.iou(location_revised, gt_location_revised)
                if ious >= max_ious:
                        max_ious = ious
                        index = id
        if max_ious != 0:
            best_location = locations[index]
            class_index = self.classes.index(best_location[0])
            best_location[0]= class_index
            return best_location
        else:   # it means the detection failed, no intersection with the ground truth
            return [0, 0, 0, 0, 0, 0]

    def location_from_0_to_1(self, wid, ht, location):
        location[1] /= wid
        location[2] /= ht
        location[3] /= wid
        location[4] /= ht
        return location

    def compute_yolo_output(self, img, gt_location):
        self.h_img, self.w_img,_ = img.shape
        img_resized = cv2.resize(img, (448, 448))
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray(img_RGB)
        inputs = np.zeros((1, 448, 448, 3), dtype = 'float32')
        inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        in_dict = {self.x : inputs}

        feature= self.sess.run(self.fc_30, feed_dict = in_dict)
        output = self.sess.run(self.fc_32, feed_dict = in_dict)  # make sure it does not run conv layers twice

        locations = self.interpret_output(output[0])
        location = self.find_best_location(locations, gt_location) # find the ROI that has the maximum IOU with the ground truth

        # change location into [0, 1]
        location = self.location_from_0_to_1(self.w_img, self.h_img, location)
        yolo_output = np.concatenate(
            (np.reshape(feature, [-1, self.num_feat]), np.reshape(location, [-1, self.num_predict])),
            axis = 1)

        return yolo_output
