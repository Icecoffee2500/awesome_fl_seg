from torch.optim.lr_scheduler import LinearLR, LambdaLR, SequentialLR

def build_epoch_scheduler(optimizer, scheduler_cfg):
    max_epochs = scheduler_cfg.max_epochs
    warmup_epochs = scheduler_cfg.warmup_epochs
    power = scheduler_cfg.power

    # 1) Warmup: Linear 증가
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    # 2) PolyLR: LambdaLR로 직접 구현
    def poly_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 1.0  # warmup에서 이미 처리했음
        factor = 1 - (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return factor ** power

    poly_scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_lambda)

    # 3) 두 scheduler 연결
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, poly_scheduler],
        milestones=[warmup_epochs]
    )
    return scheduler