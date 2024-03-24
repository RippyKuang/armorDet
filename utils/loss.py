# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

from math import sqrt
import torch
import torch.nn as nn

from utils.metrics import bbox_iou, xyxyxyxy2xyxy
from utils.torch_utils import de_parallel
import torch.nn.functional as F


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss




class ComputeKLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=True):
        device = next(model.parameters()).device  # get model device

        m = de_parallel(model).model[-1]  # Detect() module
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
  
        self.device = device
        self.clr = 3
        self.strides = [8,16,32]
        self.grids = [torch.zeros(1)]*( self.nc+3 +8+1)
        self.use_l1 = False
       
        self.l1_loss =nn.SmoothL1Loss(reduction='none')
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        
       

    def __call__(self, p, targets):  # predictions, targets
        
        x_shifts = []
        y_shifts = []
        grids = []
        expanded_strides = []
        outputs = []
        
        for i, pi in enumerate(p):  # layer index, layer predictions
            pi, grid = self.get_output_and_grid(pi, i, self.strides[i], p[0].type())
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(self.strides[i])
                    .type_as(p[0])
                )
            grids.append(grid)
            outputs.append(pi)
        return self.get_losses(
                x_shifts,
                y_shifts,
                grids,
                expanded_strides,
                targets,
                torch.cat(outputs, 1),
                dtype=p[0].dtype,
            )           
           
    
    
    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = self.nc +8+3
        hsize, wsize = output.shape[-3:-1]
        if grid.shape[2:4] != output.shape[1:3]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, 1,  hsize, wsize,n_ch).reshape(batch_size, hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output[..., :8] = (output[..., :8] + grid.repeat(1,1,4)) * stride
       
        return output, grid
    
    def get_losses(
        self,
        x_shifts,          #0.shape 1,6400 1.shape 1,1600   2.shape 1,400
        y_shifts,          #same as x_shifts
        grids,             #0.shape 1,6400,2 coord to outputs
        expanded_strides,  #8*6400,16*1600,32*400
        yolo_labels,       #abs coord
        outputs,           #abs coord
        dtype,             #torch.float32
    ):
        bbox_preds = outputs[:, :, :8]  # [batch, n_anchors_all, 4]
        clr_preds = outputs[:, :, 8:11]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 11:]  # [batch, n_anchors_all, n_cls]

     
    

        total_num_anchors = outputs.shape[1]  #6400+1600+400
        x_shifts = torch.cat(x_shifts, 1)  # [1, 8400]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        grids = torch.cat(grids, 1)
      

        cls_targets = []
        clr_targets = []
        reg_targets = []
    
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            labels = yolo_labels[yolo_labels[:,0]==batch_idx]
            num_gt = len(labels)
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.nc))
                clr_target = outputs.new_zeros((0, self.clr))
                reg_target = outputs.new_zeros((0, 8))
              
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[:, 2:] 
                gt_classes = labels[:, 1]
                bboxes_preds_per_image = bbox_preds[batch_idx]

        
                (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        clr_preds,
                )
                
                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64) // 3, self.nc
                ) * pred_ious_this_matching.unsqueeze(-1)
                clr_target = F.one_hot(
                    gt_matched_classes.to(torch.int64) % 3, self.clr
                )
         
                reg_target = gt_bboxes_per_image[matched_gt_inds]
               

            cls_targets.append(cls_target)
            clr_targets.append(clr_target)
            reg_targets.append(reg_target)

            fg_masks.append(fg_mask)
 

        _cls_targets = torch.cat(cls_targets, 0)
        _clr_targets = torch.cat(clr_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
   
        fg_masks = torch.cat(fg_masks, 0)
     
        # cls_preds = cls_preds.view(-1, self.nc)
        # clr_preds = clr_preds.view(-1, self.clr)

        # cls_targets = torch.zeros_like(cls_preds)
        # cls_targets[fg_masks] = _cls_targets
        
        # clr_targets = torch.zeros_like(clr_preds)
        # clr_targets[fg_masks] = _clr_targets
      
      
        num_fg = max(num_fg, 1)
        
        loss_iou =  (1.0 - bbox_iou(bbox_preds.view(-1, 8)[fg_masks], reg_targets,xywh=False,CIoU=True).squeeze()).sum() / num_fg
     
        loss_cls = self.bcewithlog_loss(cls_preds, cls_targets).sum() / num_fg
        loss_clr = self.bcewithlog_loss(clr_preds, clr_targets.float()).sum() / num_fg
       
        loss_l1 = (self.l1_loss(bbox_preds.view(-1, 8)[fg_masks], reg_targets)).sum() / num_fg

        reg_weight = 8.0
        l1_weight = 0.2
        loss = reg_weight * loss_iou  + loss_cls + l1_weight * loss_l1 +loss_clr

        return loss, torch.cat((reg_weight*loss_iou.reshape(1), loss_cls.reshape(1), (loss_clr).reshape(1),l1_weight * loss_l1.reshape(1))).detach()
    
    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image, #8400,8
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        clr_preds,
        ):


        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        clr_preds_ = clr_preds[batch_idx][fg_mask]
    
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]


        pair_wise_ious = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, True)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64)//3, self.nc )
            .float()
        )
        gt_clr_per_image = (
            F.one_hot(gt_classes.to(torch.int64)%3, self.clr )
            .float()
        )
 
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)


        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_()
            ).sqrt()
            clr_preds_ = (
                clr_preds_.float().sigmoid_()
            ).sqrt()

            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)

            pair_wise_clr_loss = F.binary_cross_entropy(
                clr_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_clr_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)


        del cls_preds_,clr_preds_

        cost = (
            0.5*pair_wise_cls_loss+
            0.5*pair_wise_clr_loss+
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
               )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss,pair_wise_clr_loss ,cost, pair_wise_ious, pair_wise_ious_loss



        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )
    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5)* expanded_strides_per_image).unsqueeze(0) # each anchor center coord
        y_centers_per_image = ((y_shifts[0] + 0.5)* expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 2.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius #abs radius

        center_x = torch.sum(gt_bboxes_per_image[...,[0,2,4,6]],dim=-1,keepdim=True)/4
       
        center_y = torch.sum(gt_bboxes_per_image[...,[1,3,5,7]],dim=-1,keepdim=True)/4
       

        gt_bboxes_per_image_l = center_x - center_dist
        gt_bboxes_per_image_r = center_x + center_dist #(left top ,right bottom)
        gt_bboxes_per_image_t = center_y - center_dist
        gt_bboxes_per_image_b = center_y + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image  #gt in 1.5 radius 
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0  #3,8400 
        anchor_filter = is_in_centers.sum(dim=0) > 0#1,8400 有用的anchor
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
    
    def bboxes_iou(self,bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            bboxes_a = xyxyxyxy2xyxy(bboxes_a)
            bboxes_b = xyxyxyxy2xyxy(bboxes_b)
        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
        return area_i / (area_a[:, None] + area_b - area_i)
