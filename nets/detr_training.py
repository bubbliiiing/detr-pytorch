# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from functools import partial

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from nets.ops import box_cxcywh_to_xyxy, generalized_box_iou

from . import ops
from .ops import accuracy, get_world_size, is_dist_avail_and_initialized


class HungarianMatcher(nn.Module):
    """
    此Matcher计算真实框和网络预测之间的分配
    因为预测多于目标，对最佳预测进行1对1匹配。
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        # 这是Cost中分类错误的相对权重
        self.cost_class = cost_class
        # 这是Cost中边界框坐标L1误差的相对权重
        self.cost_bbox = cost_bbox
        # 这是Cost中边界框giou损失的相对权重
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        # 获得输入的batch_size和query数量
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 将预测结果的batch维度进行平铺
        # [batch_size * num_queries, num_classes]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        # [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  

        # 将真实框进行concat
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 计算分类成本。预测越准值越小。
        cost_class = -out_prob[:, tgt_ids]

        # 计算预测框和真实框之间的L1成本。预测越准值越小。
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 计算预测框和真实框之间的IOU成本。预测越准值越小。
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        # 最终的成本矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # 对每一张图片进行指派任务，也就是找到真实框对应的num_queries里面最接近的预测结果，也就是指派num_queries里面一个预测框去预测某一个真实框
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # 返回指派的结果
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    """ 
    计算DETR的损失。该过程分为两个步骤：
    1、计算了真实框和模型输出之间的匈牙利分配
    2、根据分配结果计算损失
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        # 类别数量
        self.num_classes    = num_classes
        # 用于匹配的匹配类HungarianMatcher
        self.matcher        = matcher
        # 损失的权值分配
        self.weight_dict    = weight_dict
        # 背景的权重
        self.eos_coef       = eos_coef
        # 需要计算的损失
        self.losses         = losses
        # 种类的权重
        empty_weight        = torch.ones(self.num_classes + 1)
        empty_weight[-1]    = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        # 首先计算不属于辅助头的损失
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # 通过matcher计算每一个图片，预测框和真实框的对应情况
        indices = self.matcher(outputs_without_aux, targets)

        # 计算这个batch中所有图片的总的真实框数量
        # 计算所有节点的目标框的平均数量，以实现标准化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 计算所有的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 在辅助损失的情况下，我们对每个中间层的输出重复此过程。
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # 根据名称计算损失
        loss_map = {
            'labels'        : self.loss_labels,
            'cardinality'   : self.loss_cardinality,
            'boxes'         : self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        # 获得输出中的分类部分
        src_logits          = outputs['pred_logits']

        # 找到预测结果中有对应真实框的预测框
        idx                 = self._get_src_permutation_idx(indices)
        # 获得整个batch所有框的类别
        target_classes_o    = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes      = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        # 将其中对应的预测框设置为目标类别，否则为背景
        target_classes[idx] = target_classes_o

        # 计算交叉熵
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits     = outputs['pred_logits']
        device          = pred_logits.device
        
        # 计算每个batch真实框的数量
        tgt_lengths     = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # 计算不是背景的预测数
        card_pred       = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # 然后将不是背景的预测数和真实情况做一个l1损失
        card_err        = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses          = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        # 找到预测结果中有对应真实框的预测框
        idx             = self._get_src_permutation_idx(indices)
        # 将预测结果中有对应真实框的预测框取出
        src_boxes       = outputs['pred_boxes'][idx]
        # 取出真实框
        target_boxes    = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # 预测框和所有的真实框计算l1的损失
        loss_bbox       = F.l1_loss(src_boxes, target_boxes, reduction='none')
        # 计算giou损失
        loss_giou       = 1 - torch.diag(ops.generalized_box_iou(ops.box_cxcywh_to_xyxy(src_boxes), ops.box_cxcywh_to_xyxy(target_boxes)))
        # 返回两个损失
        losses              = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx   = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx     = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx   = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx     = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

def build_loss(num_classes, dec_layers=6, aux_loss=False):
    # 用到的真实框与预测框的匹配器
    matcher                     = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    # 不同损失的权重
    weight_dict                 = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    # TODO this is a hack
    if aux_loss:
        aux_weight_dict         = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    # 要用到的三个损失
    losses      = ['labels', 'boxes', 'cardinality']
    
    # 构建损失的类
    criterion   = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.1, losses=losses)
    return criterion

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch, lr_scale_ratio):
    lr = lr_scheduler_func(epoch)
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * lr_scale_ratio[i]