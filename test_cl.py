from pathlib import Path
import torch
from segmentation.datasets.cityscapes import CityscapesDataset
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Segformer
from utils import save_data, print_keys
from evaluate import evaluate

ROOT = Path.cwd()

@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(cfg:DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # config = OmegaConf.to_container(cfg, resolve=True) # dict로 변환 (resolve=True: 참조 해결)

    # 데이터셋 로드
    val_dataset = CityscapesDataset(
        root=cfg.dataset.data_root,
        split="val",
        mode=cfg.dataset.mode,
        target_type=cfg.dataset.target_type,
        pipeline_cfg=cfg.dataset.test_pipeline,
    )

    # 데이터로더 설정
    valid_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.val_dataloader.batch_size,
        shuffle=cfg.dataset.val_dataloader.sampler.shuffle,
        num_workers=cfg.dataset.val_dataloader.num_workers,
        pin_memory=True,
    )

    # 모델 로드
    device = f"cuda:{cfg.device_id}" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    # model.to(device)
    # checkpoint = "smp-hub/segformer-b0-1024x1024-city-160k"
    checkpoint = cfg.test_checkpoint_path
    print(f"checkpoint: {checkpoint}")
    # model = smp.from_pretrained(checkpoint).eval().to(device)

    model = Segformer(
        encoder_name="mit_b0",            # MiT-B0
        encoder_weights="imagenet",       # ImageNet pretrained
        decoder_channels=256,             # decoder 내부 채널
        in_channels=3,
        classes=19,                       # Cityscapes 클래스 수
        encoder_output_stride=32          # stage 0~3 출력 stride 설정 (smp 기본값)
    )
    model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    # model.eval()

    # evaluation ------------------------------------------------------------
    preds_dir = ROOT / "predictions"  # 예측 파일 저장 경로
    preds_dir.mkdir(parents=True, exist_ok=True)

    test_data_root = Path(cfg.test_data_root)

    evaluate(
        model=model,
        valid_loader=valid_loader,
        device=device,
        data_root=test_data_root,
        output_dir=preds_dir
    )

if __name__ == "__main__":
    main()





# 원본
# bochum_000000_000313_leftImg8bit.png

# gt
# bochum_000000_000313_gtFine_color.png # 디버깅/시각화 용
# bochum_000000_000313_gtFine_instanceIds.png # instance/panoptig seg에 사용
# bochum_000000_000313_gtFine_labelIds.png # 원래 클래스 ID로 채워진 mask # 직접 학습에 쓰려면 TrainId 체계로 매핑해줘야 함.
# bochum_000000_000313_gtFine_labelTrainIds.png # 학습용으로 미리 매핑된 TrainId 마스크 # 보통 0 ~ N-1 같은 연속적인 ID를 사용하고, 학습에서 무시할 클래스는 255로 표기.
# bochum_000000_000313_gtFine_polygons.json # 인스턴스의 폴리곤 좌표, 레이블명, 이미지 메타데이터 등이 들어있음.
