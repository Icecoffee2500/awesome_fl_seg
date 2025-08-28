import torch

def get_param_groups(model, base_lr, weight_decay, paramwise_cfg):
    param_groups = []
    for name, param in model.named_parameters():
        # print(f"name: {name} | param.shape: {param.shape} | param.requires_grad: {param.requires_grad}")
        if not param.requires_grad:
            continue

        # 기본 그룹 설정
        group = {"params": [param], "lr": base_lr, "weight_decay": weight_decay}

        # 1) pos_block → decay_mult=0
        if "pos_block" in name:
            group["weight_decay"] = paramwise_cfg.pos_block.weight_decay

        # 2) norm → decay_mult=0
        elif "norm" in name or "bn" in name or "layernorm" in name:  
            # SegFormer는 LayerNorm을 많이 쓰니까 layernorm도 체크
            group["weight_decay"] = paramwise_cfg.norm.weight_decay

        # 3) head → lr_mult=10
        elif "head" in name or "decode_head" in name:
            group["lr"] = base_lr * paramwise_cfg.head.lr_mult

        param_groups.append(group)

    return param_groups

def build_optimizer(model, optimizer_cfg):
    base_lr = optimizer_cfg.lr
    betas = optimizer_cfg.betas
    weight_decay = optimizer_cfg.weight_decay

    param_groups = get_param_groups(model, base_lr, weight_decay, optimizer_cfg.paramwise_cfg)
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=betas, weight_decay=weight_decay)
    return optimizer