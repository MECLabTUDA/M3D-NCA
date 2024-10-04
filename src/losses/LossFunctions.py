import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    r"""Dice Loss
    """
    def __init__(self, useSigmoid: bool = True) -> None:
        r"""Initialisation method of DiceLoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceLoss, self).__init__()

    def forward(self, input: torch.tensor, target: torch.tensor, smooth: float = 1) -> torch.Tensor:
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
    def __init__(self, useSigmoid = True) -> None:
        r"""Initialisation method of DiceLoss mask
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceLoss_mask, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, smooth: float=1) -> torch.Tensor:
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
    def __init__(self, useSigmoid: bool = True) -> None:
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceBCELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1) -> torch.Tensor:
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
    def __init__(self, useSigmoid: bool = True) -> None:
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(BCELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1) -> torch.Tensor:
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
    def __init__(self, gamma: float = 2, eps: float = 1e-7) -> None:
        r"""Initialisation method of DiceBCELoss
            #Args:
                gamma
                eps
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, gamma: float = 2, eps: float = 1e-7):
        r"""Initialisation method of DiceBCELoss
            #Args:
                gamma
                eps
        """
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
    
class WeightedDiceBCELoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        input = torch.sigmoid(input)
        input_flat = torch.flatten(input)
        target_flat = torch.flatten(target)
        variance_flat = 1 - 2*torch.flatten(variance)  # Assuming variance is already prepared for weighting

        # Weighted intersection for Dice
        weighted_intersection = (input_flat * target_flat * variance_flat).sum()
        weighted_input_sum = (input_flat * variance_flat).sum()
        weighted_target_sum = (target_flat * variance_flat).sum()
        
        dice_loss = 1 - (2. * weighted_intersection + 1.) / (weighted_input_sum + weighted_target_sum + 1.)

        # BCE Loss with variance weighting
        bce_loss = torch.nn.functional.binary_cross_entropy(input_flat, target_flat, weight=variance_flat, reduction='mean')

        # Combining Dice and BCE losses
        total_loss = bce_loss + dice_loss

        return total_loss
    


class DiceFocalLoss_2(FocalLoss):
    r"""Dice Focal Loss
    """
    def __init__(self, gamma: float = 2, eps: float = 1e-7):
        r"""Initialisation method of DiceBCELoss
            #Args:
                gamma
                eps
        """
        super(DiceFocalLoss_2, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Forward function
            #Args:
                input: input array
                target: target array
        """
        input = torch.sigmoid(input)
        target = torch.sigmoid(target).detach()
        target = torch.round(target)
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