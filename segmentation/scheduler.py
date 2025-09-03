from torch.optim.lr_scheduler import LinearLR, LambdaLR, SequentialLR
import torch

# print(f"torch version: {torch.__version__}")

def build_epoch_scheduler(optimizer, scheduler_cfg):
    max_epochs = scheduler_cfg.max_epochs
    warmup_epochs = scheduler_cfg.warmup_epochs
    power = scheduler_cfg.power
    min_lr = scheduler_cfg.min_lr

    # 1) PolyLR: LambdaLR로 구현 (warmup 후 poly decay)
    def poly_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 1.0  # Warmup 기간 동안은 poly가 적용되지 않음 (SequentialLR로 분리)
        
        # Poly factor 계산
        decay_epochs = epoch - warmup_epochs
        total_decay_epochs = max_epochs - warmup_epochs
        if decay_epochs >= total_decay_epochs:
            return min_lr / optimizer.param_groups[0]['lr']  # base_lr 기준으로 normalize (but since min_lr=0, it's 0)
        
        # power=1.0이기 때문에 사실상 Linear decay이다.
        factor = (1 - (decay_epochs / total_decay_epochs)) ** power
        # min_lr 적용: min_lr + (base_lr - min_lr) * factor (but base_lr is normalized to 1 in lambda)
        return factor  # LambdaLR은 base_lr에 곱하므로, min_lr=0이라 factor만 반환 (필요시 조정)

    poly_scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_lambda)

    # 2) Warmup: Linear 증가
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1,
        total_iters=warmup_epochs,
        # verbose=True
    )
    # learning rate 그래프 그려보기
    # graph calculator <- 그려봐야 함.

    # 3) 두 scheduler 연결
    # milestones 전에는 첫번째 scheduler로 step() 호출.
    # milestones에 도달하면 두번째 scheduler로 step(0) 호출. -> epoch=0이라는 것을 명시해줌. -> self.last_epoch=0이 됨.
    # self._last_lr도 idx에 맞는 scheduler의 get_last_lr() 결과를 반환.
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, poly_scheduler],
        milestones=[warmup_epochs]
    )
    return scheduler
    # return warmup_scheduler