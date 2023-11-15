from ..builder import PIPELINES
import numpy as np
import random


@PIPELINES.register_module('SmallObjectAugmentation')
class SmallObjectAugmentation:

    def __init__(self, thresh=64*64, prob=0.5, copy_times=3, epochs=30, all_objects=False, one_object=False):
        '''
        SmallObjectAugmentation: https://arxiv.org/abs/1902.07296
        https://github.com/zzl-pointcloud/Data_Augmentation_Zoo_for_Object_Detection/blob/master/augmentation_zoo/SmallObjectAugmentation.py
        args:
            thresh: small object thresh
            prob: the probability of whether to augmentation
            copy_times: how many times to copy anno
            epochs: how many times try to create anno
            all_object: copy all object once
            one_object: copy one object
        '''
        self.thresh = thresh
        self.prob = prob
        self.copy_times = copy_times
        self.epochs = epochs
        self.all_objects = all_objects
        self.one_object = one_object
        if self.all_objects or self.one_object:
            self.copy_times = 1

    def _is_small_object(self, height, width):
        '''
        判断是否为小目标
        '''
        if height*width <= self.thresh:
            return True
        else:
            return False

    def _compute_overlap(self, bbox_a, bbox_b):
        '''
        计算重叠
        '''
        if bbox_a is None:
            return False
        left_max = max(bbox_a[0], bbox_b[0])
        top_max = max(bbox_a[1], bbox_b[1])
        right_min = min(bbox_a[2], bbox_b[2])
        bottom_min = min(bbox_a[3], bbox_b[3])
        inter = max(0, (right_min-left_max)) * max(0, (bottom_min-top_max))
        if inter != 0:
            return True
        else:
            return False

    def _donot_overlap(self, new_bbox, bboxes):
        '''
        是否有重叠
        '''
        for bbox in bboxes:
            if self._compute_overlap(new_bbox, bbox):
                return False
        return True

    def _create_copy_annot(self, height, width, bbox, bboxes):
        '''
        创建新的标签
        '''
        bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        for epoch in range(self.epochs):
            random_x, random_y = np.random.randint(int(bbox_w / 2), int(width - bbox_w / 2)), \
                np.random.randint(int(bbox_h / 2), int(height - bbox_h / 2))
            tl_x, tl_y = random_x - bbox_w/2, random_y-bbox_h/2
            br_x, br_y = tl_x + bbox_w, tl_y + bbox_h
            if tl_x < 0 or br_x > width or tl_y < 0 or tl_y > height:
                continue
            new_bbox = np.array([tl_x, tl_y, br_x, br_y], dtype=np.int32)

            if not self._donot_overlap(new_bbox, bboxes):
                continue

            return new_bbox
        return None

    def _add_patch_in_img(self, new_bbox, copy_bbox, image):
        '''
        复制图像区域
        '''
        copy_bbox = copy_bbox.astype(np.int32)

        image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2],
              :] = image[copy_bbox[1]:copy_bbox[3], copy_bbox[0]:copy_bbox[2], :]
        return image

    def __call__(self, results):
        if self.all_objects and self.one_object:
            return results
        if np.random.rand() > self.prob:
            return results

        img = results['img']
        bboxes = results['gt_bboxes']
        labels = results['gt_labels']

        height, width = img.shape[0], img.shape[1]

        small_object_list = []
        for idx in range(len(bboxes)):
            bbox = bboxes[idx]
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if self._is_small_object(bbox_h, bbox_w):
                small_object_list.append(idx)

        length = len(small_object_list)
        # 无小物体
        if 0 == length:
            return results

        # 随机选择不同的物体复制
        copy_object_num = np.random.randint(0, length)
        if self.all_objects:
            # 复制全部物体
            copy_object_num = length
        if self.one_object:
            # 只选择一个物体复制
            copy_object_num = 1

        random_list = random.sample(range(length), copy_object_num)
        idx_of_small_objects = [small_object_list[idx] for idx in random_list]
        select_bboxes = bboxes[idx_of_small_objects, :]
        select_labels = labels[idx_of_small_objects]

        bboxes = bboxes.tolist()
        labels = labels.tolist()
        for idx in range(copy_object_num):
            bbox = select_bboxes[idx]
            label = select_labels[idx]

            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if not self._is_small_object(bbox_h, bbox_w):
                continue

            for i in range(self.copy_times):
                new_bbox = self._create_copy_annot(height, width, bbox, bboxes)
                if new_bbox is not None:
                    img = self._add_patch_in_img(new_bbox, bbox, img)
                    bboxes.append(new_bbox)
                    labels.append(label)
                
        
        results['img'] = img
        results['gt_bboxes'] = np.array(bboxes)
        results['gt_labels'] = np.array(labels)
        return results
