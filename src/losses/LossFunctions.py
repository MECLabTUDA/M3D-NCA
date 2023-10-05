import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    r"""Dice Loss
    """
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceLoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceLoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        if self.useSigmoid:
            input = torch.sigmoid(input)  
        input = torch.flatten(input)
        target = torch.flatten(target)
        intersection = (input * target).sum()
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)

        return 1 - dice

class DiceLoss_mask(torch.nn.Module):
    r"""Dice Loss mask, that only calculates on masked values
    """
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceLoss mask
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceLoss_mask, self).__init__()

    def forward(self, input, target, mask = None, smooth=1):
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
                mask: The mask which defines which values to consider
        """
        if self.useSigmoid:
            input = torch.sigmoid(input)  
        input = torch.flatten(input)
        target = torch.flatten(target)
        mask = torch.flatten(mask)

        input = input[~mask]  
        target = target[~mask]  
        intersection = (input * target).sum()
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)

        return 1 - dice

class DiceBCELoss(torch.nn.Module):
    r"""Dice BCE Loss
    """
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceBCELoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        input = torch.sigmoid(input)       
        input = torch.flatten(input) 
        target = torch.flatten(target)
        
        intersection = (input * target).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)  
        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class BCELoss(torch.nn.Module):
    r"""BCE Loss
    """
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(BCELoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        input = torch.sigmoid(input)       
        input = torch.flatten(input) 
        target = torch.flatten(target)

        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        return BCE

class FocalLoss(torch.nn.Module):
    r"""Focal Loss
    """
    def __init__(self, gamma=2, eps=1e-7):
        r"""Initialisation method of DiceBCELoss
            #Args:
                gamma
                eps
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        r"""Forward function
            #Args:
                input: input array
                target: target array
        """
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_bce = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        loss = loss_bce * (1 - logit) ** self.gamma  # focal loss
        loss = loss.mean()
        return loss


class DiceFocalLoss(FocalLoss):
    r"""Dice Focal Loss
    """
    def __init__(self, gamma=2, eps=1e-7):
        r"""Initialisation method of DiceBCELoss
            #Args:
                gamma
                eps
        """
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        r"""Forward function
            #Args:
                input: input array
                target: target array
        """
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        intersection = (input * target).sum()
        dice_loss = 1 - (2.*intersection + 1.)/(input.sum() + target.sum() + 1.)

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_bce = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        focal = loss_bce * (1 - logit) ** self.gamma  # focal loss
        dice_focal = focal.mean() + dice_loss
        return dice_focal