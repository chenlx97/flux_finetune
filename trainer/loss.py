import torch

def compute_flow_matching_loss(model_pred, noise, model_input, sigmas):
    target = noise - model_input
    weighting = torch.ones_like(target) # 简化权重，原脚本有 compute_loss_weighting_for_sd3
    
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )
    return loss.mean()