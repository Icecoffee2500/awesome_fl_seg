from pathlib import Path
import torch
from segmentation.datasets.cityscapes import CityscapesDataset
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
# import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Segformer
from utils import save_data, print_keys
from evaluate import evaluate
from segmentation.optimizer import build_optimizer
from segmentation.scheduler import build_epoch_scheduler

ROOT = Path.cwd()

def train_one_epoch(model, optimizer, criterion, train_loader, device, epoch):
    model.train()
    epoch_loss = 0.0

    for idx, (inputs, masks, img_infos) in enumerate(train_loader):
        inputs = inputs.to(device)
        # masks = masks.to(device)
        masks = masks.to(device).squeeze(1)  # [B, H, W]

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if idx % 50 == 0:
            print(f"Epoch [{epoch+1}/10][{idx}/{len(train_loader)}]: Loss: {loss.item():.4f}")
    
    return epoch_loss / len(train_loader)

@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(cfg:DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # config = OmegaConf.to_container(cfg, resolve=True) # dict로 변환 (resolve=True: 참조 해결)

    # 데이터셋 로드
    train_dataset = CityscapesDataset(
        root=cfg.dataset.data_root,
        split="train",
        mode=cfg.dataset.mode,
        target_type=cfg.dataset.target_type,
        pipeline_cfg=cfg.dataset.train_pipeline,
    )
    
    val_dataset = CityscapesDataset(
        root=cfg.dataset.data_root,
        split="val",
        mode=cfg.dataset.mode,
        target_type=cfg.dataset.target_type,
        pipeline_cfg=cfg.dataset.test_pipeline,
    )

    # 데이터로더 설정
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.train_dataloader.batch_size,
        shuffle=cfg.dataset.train_dataloader.sampler.shuffle,
        num_workers=cfg.dataset.train_dataloader.num_workers,
        pin_memory=True,
    )
    
    valid_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.val_dataloader.batch_size,
        shuffle=cfg.dataset.val_dataloader.sampler.shuffle,
        num_workers=cfg.dataset.val_dataloader.num_workers,
        pin_memory=True,
    )

    # 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # checkpoint = "smp-hub/segformer-b0-1024x1024-city-160k"
    # model = smp.from_pretrained(checkpoint).to(device)
    
    model = Segformer(
        encoder_name="mit_b0",            # MiT-B0
        encoder_weights="imagenet",       # ImageNet pretrained
        decoder_channels=256,             # decoder 내부 채널
        in_channels=3,
        classes=19,                       # Cityscapes 클래스 수
        encoder_output_stride=32          # stage 0~3 출력 stride 설정 (smp 기본값)
    )
    model.to(device)

    # optimizer, scheduler, criterion 설정
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_epoch_scheduler(optimizer, cfg.scheduler)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # evaluation
    preds_dir = ROOT / "predictions"  # 예측 파일 저장 경로
    preds_dir.mkdir(parents=True, exist_ok=True)

    # train
    for epoch in range(cfg.trainer.epochs):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device, epoch)
        print(f"Epoch [{epoch+1}/{cfg.trainer.epochs}]: Loss: {train_loss:.4f}")
        scheduler.step()

        # evaluation
        if epoch % cfg.trainer.eval_interval == 0:
            evaluate(model, valid_loader, device, data_root=Path(cfg.dataset.data_root), output_dir=preds_dir)

if __name__ == "__main__":
    main()

