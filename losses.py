import torch.nn as nn
import torch
import torch.nn.functional as F

class WeightedMultiNCELoss(nn.Module):
    def __init__(self,
                 output_fn,
                 temperature=0.1,
                 out=True,
                 **kwargs):
        super(WeightedMultiNCELoss, self).__init__()
        self.temperature = temperature
        self.out = out
        self.output_fn = output_fn

    def _calculate_loss(self, scores, masks):
        num_pos = masks.sum(1)
        if not self.out:
            loss = - torch.log(
                (F.softmax(scores / self.temperature, dim=1) * masks).sum(1) / num_pos)
        else:
            loss = - (torch.log(
                (F.softmax(scores / self.temperature, dim=1))) * masks).sum(1) / num_pos
        return loss.mean()

    def forward(self, model_output):
        scores, masks = self.output_fn(model_output)
        losses = 0
        if isinstance(scores, list):
            for score, mask in zip(scores, masks):
                losses += self._calculate_loss(score, mask)
            return losses
        else:
            return self._calculate_loss(scores, masks)

    
class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, model_output):
        logits, targets = model_output['logit'], model_output['target']
        return self.loss(logits, targets)

class WeightedRankPairwiseLoss(nn.Module):
    def __init__(self, margin=1, reg_lambda=1, **kwargs):
        super(WeightedRankPairwiseLoss, self).__init__()
        self.margin = margin
        self.reg_lambda = reg_lambda
    
    @torch.no_grad()
    def _get_weight(self, ranks):
        # ranks: (B,)
        def _func(rank):
            w = 0.0
            for i in range(1, int(rank) + 1):
                w += 1 / i
            return w
        
        weights = ranks.clone().to(torch.float).cpu().apply_(_func)
        return weights.to(ranks.device)
    
    def _hinge_loss(self, scores, targets):
        """
        calculate \sum_y l(x_n, y_n, y)
            - l(x_n, y_n, y) = margin + s(x_n, y) - s(x_n, y_n)
            - margin: equals to margin if y_n != y else 0
            - y_n is the ground truth
            - s(x_n, y): score between audio embedding x_n and text embedding y
        """
        masks = targets.unsqueeze(-1) != torch.arange(scores.shape[-1], device=targets.device).unsqueeze(0)
        margins = masks * self.margin
        gt_scores = torch.gather(scores, dim=1, index=targets.unsqueeze(-1)) # (B, 1)
        loss = (margins + scores - gt_scores) * masks
        mask_positive = (loss > 0).to(loss.dtype)
        loss *= mask_positive
        return loss.sum(dim=-1), mask_positive.sum(dim=-1) # (B,)
 
    def forward(self, model_output):
        scores, targets, W = model_output['score'],\
                             model_output['target'],\
                             model_output.get('W', None)
        sorted_indices = torch.argsort(scores, dim=-1, descending=True)
        rank_of_gt = torch.argmax((sorted_indices == targets.unsqueeze(-1)).to(torch.int), dim=-1) + 1
        hinge_loss, n_pos = self._hinge_loss(scores, targets) # (B,)

        rank = rank_of_gt 
        rank_weights = self._get_weight(ranks=rank) # (B,)

        n_pos += 1e-7 # to prevent 0-divizor
        loss = (rank_weights / n_pos) * hinge_loss
        if W is not None:
            param_norm = 0
            if isinstance(W, list):
                for param in W:
                    param_norm += param.norm() ** 2
            else:
                param_norm = W.norm() ** 2
            return loss.mean() + self.reg_lambda * param_norm
        else:
            return loss.mean()


class WeightedRankPairwiseLoss2(nn.Module):
    def __init__(self, margin=1, reg_lambda=1, **kwargs):
        super(WeightedRankPairwiseLoss2, self).__init__()
        self.margin = margin
        self.reg_lambda = reg_lambda
    
    @torch.no_grad()
    def _get_weight(self, ranks):
        # ranks: (B,)
        def _func(rank):
            w = 0.0
            for i in range(1, int(rank) + 1):
                w += 1 / i
            return w
        
        weights = ranks.clone().to(torch.float).cpu().apply_(_func)
        return weights.to(ranks.device)
    
    def _hinge_loss(self, scores, targets):
        """
        calculate \sum_y l(x_n, y_n, y)
            - l(x_n, y_n, y) = margin + s(x_n, y) - s(x_n, y_n)
            - margin: equals to margin if y_n != y else 0
            - y_n is the ground truth
            - s(x_n, y): score between audio embedding x_n and text embedding y
        """
        masks = targets.unsqueeze(-1) != torch.arange(scores.shape[-1], device=targets.device).unsqueeze(0)
        margins = masks * self.margin
        gt_scores = torch.gather(scores, dim=1, index=targets.unsqueeze(-1)) # (B, 1)
        loss = (margins + scores - gt_scores) * masks
        mask_positive = (loss > 0).to(loss.dtype)
        loss *= mask_positive
        return loss.sum(dim=-1), mask_positive.sum(dim=-1) # (B,)
 
    def forward(self, model_output):
        scores, targets, W = model_output['score'],\
                             model_output['target'],\
                             model_output.get('W', None)
        sorted_indices = torch.argsort(scores, dim=-1, descending=True)
        rank_of_gt = torch.argmax((sorted_indices == targets.unsqueeze(-1)).to(torch.int), dim=-1)
        hinge_loss, _ = self._hinge_loss(scores, targets) # (B,)

        # rank = n_pos.clamp_min(1)
        rank = rank_of_gt 
        rank_weights = self._get_weight(ranks=rank) # (B,)

        weight = torch.nan_to_num(rank_weights / rank, nan=0.0)
        # convert beta/rank to 0 if rank is 0
        loss = weight * hinge_loss

        if W is not None:
            param_norm = 0
            if isinstance(W, list):
                for param in W:
                    param_norm += param.norm() ** 2
            else:
                param_norm = W.norm() ** 2
            return loss.mean() + self.reg_lambda * param_norm
        else:
            return loss.mean()