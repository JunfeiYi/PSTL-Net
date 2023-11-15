import torch
import torch.nn as nn


from ..builder import LOSSES
import numpy as np


def final_nms_loss(pos_inds, pos_gt_index, gt_bboxes, bbox_preds, cls_scores, gt_labels, pull_weight, push_weight, nms_thr, use_score,
                   add_gt, pull_relax, push_relax, push_select, fix_push_score, fix_push_reg, fix_pull_score,
                   fix_pull_reg):
    assert len(pos_inds) > 0
    img_num = len(pos_inds)
    push_loss = 0
    pull_loss = 0

    for (img_pos_inds, img_pos_gt_index, img_gt_bboxes, img_bbox_preds, img_cls_scores, img_gt_labels) in zip( pos_inds, pos_gt_index, gt_bboxes, bbox_preds, cls_scores, gt_labels,):
        single_img_loss = single_nms_loss(img_pos_inds, img_pos_gt_index, img_gt_bboxes, img_bbox_preds, img_cls_scores, img_gt_labels, nms_thr,
                                          use_score, add_gt, pull_relax, push_relax, push_select, fix_push_score,
                                          fix_push_reg, fix_pull_score, fix_pull_reg)
        push_loss = push_loss + single_img_loss['nms_push_loss']
        pull_loss = pull_loss + single_img_loss['nms_pull_loss']
    push_loss = push_loss / img_num
    pull_loss = pull_loss / img_num
    return {'nms_push_loss': push_loss * push_weight, 'nms_pull_loss': pull_loss * pull_weight}


def single_nms_loss(gt_inds, gt_index, gt_box, bbox_preds, cls_scores, gt_labels, nms_thr, use_score, add_gt, pull_relax, push_relax,
                    push_select, fix_push_score, fix_push_reg, fix_pull_score, fix_pull_reg):
    # print(torch.sum(gt_inds - anchor_gt_inds))
    # use anchor_gt_inds instead of gt_inds


    eps = 1e-6
    tmp_zero = torch.mean(gt_box).float() * 0  # used for return zero
    total_pull_loss = 0
    total_push_loss = 0
    pull_cnt = 0
    push_cnt = 0

    # discard negative proposals
    # print(gt_inds)
    # pos_mask = gt_inds >= 0  # -2:ignore, -1:negative
    if torch.sum(gt_inds) <= 1:  # when there is no positive or only one
        return {'nms_push_loss': tmp_zero, 'nms_pull_loss': tmp_zero}  # return 0
    # gt_inds = gt_inds[pos_mask]
    # proposals = proposals[pos_mask]
    labels = gt_labels[gt_index].reshape(-1, 1)
    cls_scores = cls_scores[gt_inds]
    t = []
    for i, x in enumerate(cls_scores):
        t.append(x[labels[i]])
    cls_scores = torch.stack(t)
    proposals = torch.cat([bbox_preds, cls_scores], dim=1)
    


    # perform nms
    scores = proposals[:, 4]
    v, idx = scores.sort(0)  # sort in ascending order
    if not push_select:
        iou = bbox_overlaps(proposals[:, :4], proposals[:, :4])
    else:
        # pay attention here
        # every col has gradient for the col index proposal
        # every row doesn`t have gradient for the row index proposal
        no_gradient_proposals = proposals.detach()
        iou = bbox_overlaps(no_gradient_proposals[:, :4], proposals[:, :4])

    gt_iou = bbox_overlaps(gt_box, gt_box)
    max_score_box_rec = dict()
    while idx.numel() > 0:
        # print(idx)
        i = idx[-1]  # index of current largest val
        idx = idx[:-1]  # remove kept element from view
        # cacu pull loss:
        i_gt_inds = gt_index[i]
        # print('i_gt_inds', i_gt_inds)
        i_gt_inds_value = i_gt_inds.item()
        if i_gt_inds_value in max_score_box_rec.keys():
            max_score_idx = max_score_box_rec[i_gt_inds_value]
            max_s_iou = iou[max_score_idx][i].clamp(min=eps)
            if not pull_relax:
                pull_loss = -(max_s_iou).log()
            else:
                pull_loss = -(1 - nms_thr + max_s_iou).log()
            if fix_pull_reg:
                pull_loss = pull_loss.detach()
            if use_score:
                if fix_pull_score:
                    pull_loss = pull_loss * proposals[i, 4].detach()
                else:
                    pull_loss = pull_loss * proposals[i, 4]
            pull_cnt += 1
        else:
            max_score_box_rec[i_gt_inds_value] = i
            pull_loss = tmp_zero
        # print(max_score_box_rec)
        if len(idx) == 0:
            break
        cur_iou = iou[i][idx]
        overlap_idx = cur_iou > nms_thr
        overlap_cur_iou = cur_iou[overlap_idx]
        overlap_idx_idx = idx[overlap_idx]
        cur_gt_inds = gt_index[overlap_idx_idx]
        # print('cur_gt_inds', cur_gt_inds)
        # print('i_centre', [(proposals[i,1] + proposals[i,3]) * 0.5, (proposals[i,0] + proposals[i,2]) * 0.5])
        # print('cur_centre', [(proposals[overlap_idx_idx,1] + proposals[overlap_idx_idx,3]) * 0.5, (proposals[overlap_idx_idx,0] + proposals[overlap_idx_idx,2]) * 0.5])
        cur_scores = scores[overlap_idx_idx]
        if fix_push_score:
            cur_scores = cur_scores.detach()
        # cacu push loss
        push_mask = cur_gt_inds != i_gt_inds
        # check if 0
        if torch.sum(push_mask) != 0:
            cur_gt_iou = gt_iou[i_gt_inds][cur_gt_inds]
            if not push_relax:
                push_loss = -(1 - overlap_cur_iou).log()
            else:
                push_loss = -(1 + nms_thr - overlap_cur_iou).log()
            if fix_push_reg:
                push_loss = push_loss.detach()
            # push_loss = overlap_cur_iou
            if use_score:
                push_loss = push_loss * cur_scores
            push_mask = push_mask & (overlap_cur_iou > cur_gt_iou)
            push_loss = push_loss[push_mask]
            if torch.sum(push_mask) != 0:
                push_loss = torch.mean(push_loss)
                push_cnt += int(torch.sum(push_mask).data)
            else:
                push_loss = tmp_zero
        else:
            push_loss = tmp_zero
        # print('pull_loss', pull_loss)
        # print('push_loss', push_loss)
        total_pull_loss = total_pull_loss + pull_loss
        total_push_loss = total_push_loss + push_loss
        # remove idx
        idx = idx[~overlap_idx]

    pull = total_pull_loss / (pull_cnt + eps)
    push = total_push_loss / (push_cnt + eps)
    return {'nms_push_loss': push, 'nms_pull_loss': pull}


@LOSSES.register_module()
class FinalNMSLoss(nn.Module):

    def __init__(self, reduction='none', loss_weight=1.0, use_score=True, add_gt=False, pull_relax=True,
                 push_relax=False, push_select=True, fix_push_score=False, fix_push_reg=False, fix_pull_score=False,
                 fix_pull_reg=False, pull_weight=1, push_weight=1, nms_thr=0.5):
        super(FinalNMSLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.nms_thr = nms_thr
        self.use_score = use_score
        self.add_gt = add_gt
        self.pull_relax = pull_relax
        self.push_relax = push_relax
        self.push_select = push_select
        self.fix_push_score = fix_push_score
        self.fix_push_reg = fix_push_reg
        self.fix_pull_score = fix_pull_score
        self.fix_pull_reg = fix_pull_reg

    def forward(self,
                pos_inds,
                pos_gt_index,
                gt_bboxes,
                bbox_preds,
                cls_scores,
                gt_labels,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert weight is None or weight == 1, 'please use pull/push_weight instead'
        loss_nms = final_nms_loss(
            pos_inds, pos_gt_index, gt_bboxes, bbox_preds, cls_scores, gt_labels,
            self.pull_weight, self.push_weight,
            self.nms_thr, self.use_score, self.add_gt,
            self.pull_relax, self.push_relax, self.push_select,
            self.fix_push_score, self.fix_push_reg,
            self.fix_pull_score, self.fix_pull_reg,
            **kwargs)
        return loss_nms




def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious