import torch
import torch.distributed as dist
import copy
from xtuner.parallel.sequence import get_sequence_parallel_group


class _ReduceLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean_loss, loss_scale, process_group):
        ctx.process_group = process_group
        ctx.loss_scale = copy.deepcopy(loss_scale)
        if loss_scale == 0:
            # convert nan to 0 just for logging
            mean_loss = torch.nan_to_num(mean_loss)
        loss_sum = mean_loss * loss_scale
        dist.all_reduce(loss_sum, group=process_group)
        dist.all_reduce(loss_scale, group=process_group)
        loss = loss_sum / loss_scale
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        loss_scale = ctx.loss_scale
        grad_sum = grad_output * loss_scale
        dist.all_reduce(grad_sum, group=ctx.process_group)
        dist.all_reduce(loss_scale, group=ctx.process_group)
        grad_output = grad_sum / loss_scale
        return grad_output, None, None

def reduce_sequence_parallel_loss(mean_loss,
                                  loss_scale,
                                  sp_group: dist.ProcessGroup = None):
    if dist.get_world_size(sp_group) == 1:
        return mean_loss
    if sp_group is None:
        # avoid bc breaking
        sp_group = get_sequence_parallel_group()
    return _ReduceLoss.apply(mean_loss, loss_scale, sp_group)
