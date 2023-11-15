_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

load_from='checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
