from pathlib import Path
import torch
from segmentation.datasets.cityscapes import CityscapesDataset
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
# import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Segformer
from utils import save_data, print_keys, set_seed
from evaluate import evaluate, evaluate_only_car
from segmentation.optimizer import build_optimizer
from segmentation.scheduler import build_epoch_scheduler
import wandb
from datetime import datetime
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data._utils.collate import default_collate
from PIL import Image
import numpy as np


ROOT = Path.cwd()

def custom_collate(batch):
    # batch: list of tuples returned by __getitem__()
    # 예: [(input, target, image_meta), ...]
    inputs, targets, metas = zip(*batch)  # 각각 길이=batch_size 튜플
    # 기본적으로 tensor/ndarray들만 default_collate로 병합
    batched_inputs = default_collate(inputs)
    batched_targets = default_collate(targets)
    # metas는 variable-key / variable-length일 수 있으니 리스트로 그대로 둠
    return batched_inputs, batched_targets, list(metas)

def train_one_epoch(model, optimizer, criterion, train_loader, device, curr_epoch, max_epochs):
    """
    img_infos: list (길이는 batch size)
    img_info[0]: dict(city, file_name, bboxes)
    bboxes: list of [id, x1, y1, x2, y2] # 첫번째 값은 id(class) 값!!
    """
    model.train()
    epoch_loss = 0.0
    iter_start_time = datetime.now()

    for idx, (inputs, masks, img_infos) in enumerate(train_loader):
        # save_data(inputs, masks, root_dir=ROOT)
        # print(f"[city: {img_infos['city']}] [img_filename: {img_infos['file_name']}] img_infos['bboxes']: {img_infos['bboxes']}")
        # print(f"[img_infos]:\n{img_infos}")
        # print(f"img_infos length: {len(img_infos)}")
        # print(f"type of img_infos: {type(img_infos)}")
        
        # for img_info in img_infos:
        #     print(f"[city: {img_info['city']}] [img_filename: {img_info['file_name']}] img_info['bboxes']: {img_info['bboxes']}")
        # return
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
            elapsed = datetime.now() - iter_start_time
            minutes, seconds = divmod(elapsed.total_seconds(), 60)
            print(f"Epoch [{curr_epoch+1}/{max_epochs}][{idx}/{len(train_loader)}]: "
                f"Current Loss: {loss.item():.4f} | Elapsed Time: {int(minutes)}m {seconds:.2f}s")

            # print(f"Epoch [{curr_epoch+1}/{max_epochs}][{idx}/{len(train_loader)}]: Loss: {loss.item():.4f} | Time: {datetime.now() - epoch_start_time:.2f}s")
            iter_start_time = datetime.now()
    
    # print(f"epoch_loss: {epoch_loss}")
    # print(f"len(train_loader): {len(train_loader)}")
    
    return epoch_loss / len(train_loader)

@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(cfg:DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # config = OmegaConf.to_container(cfg, resolve=True) # dict로 변환 (resolve=True: 참조 해결)
    
    now = datetime.now()
    today = now.strftime("%m%d_%H:%M")
    
    name = "train_cl_"
    name += f"{cfg.dataset.name}_crop_size-{cfg.dataset.crop_size}_gpu-{cfg.device_id}"
    name += f"_{today}"
    wdb = wandb
    wdb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        project="Segmentation training (CL)",
        name=name,
    )

    # Set random seed for reproducibility from the main entrypoint
    seed = 42
    set_seed(seed)

    # 데이터셋 로드
    train_dataset = CityscapesDataset(
        root=cfg.dataset.data_root,
        split="train",
        mode=cfg.dataset.mode,
        target_type=cfg.dataset.target_type,
        pipeline_cfg=cfg.dataset.train_pipeline,
    )
    print(f"len(train_dataset): {len(train_dataset)}")
    label_id = 26 # car class id
    # print(f"train_dataset.target_file_paths: {train_dataset.target_file_paths}")
    mask_paths = [path for path in train_dataset.target_file_paths if path.exists()]
    # print(f"len(mask_paths): {len(mask_paths)}")
    indices_with_class = []
    for i, path in enumerate(mask_paths):
        mask = np.array(Image.open(path)) # (1024, 2048)
        if np.any(mask == label_id):
            indices_with_class.append(i)

    train_dataset_with_one_class = torch.utils.data.Subset(train_dataset, indices_with_class)
    print(f"len(train_dataset_with_one_class): {len(train_dataset_with_one_class)}")
    
    val_dataset = CityscapesDataset(
        root=cfg.dataset.data_root,
        split="val",
        mode=cfg.dataset.mode,
        target_type=cfg.dataset.target_type,
        pipeline_cfg=cfg.dataset.test_pipeline,
    )
    print(f"len(val_dataset): {len(val_dataset)}")
    
    # 데이터로더 설정
    train_loader = DataLoader(
        # train_dataset,
        train_dataset_with_one_class,
        batch_size=cfg.dataset.train_dataloader.batch_size,
        shuffle=cfg.dataset.train_dataloader.sampler.shuffle,
        num_workers=cfg.dataset.train_dataloader.num_workers,
        pin_memory=True,
        # collate_fn=custom_collate,
    )
    
    valid_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.val_dataloader.batch_size,
        shuffle=cfg.dataset.val_dataloader.sampler.shuffle,
        num_workers=cfg.dataset.val_dataloader.num_workers,
        pin_memory=True,
        # collate_fn=custom_collate,
    )

    # 모델 로드
    device = f"cuda:{cfg.device_id}" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
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

    # scheduler에서 optimizer를 wrapping하고, 위의 optimizer도 같은 id를 참조하고 있기 때문에 위의 optimizer도 wrapping된 상태가 된다.
    scheduler = build_epoch_scheduler(optimizer, cfg.scheduler)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # evaluation
    # preds_dir = ROOT / "predictions"  # 예측 파일 저장 경로
    preds_dir = ROOT / f"predictions_{cfg.device_id}"  # 예측 파일 저장 경로
    preds_dir.mkdir(parents=True, exist_ok=True)

    best_performance = 0.0
    perf_dir = ROOT / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    subdir_name = f"{cfg.dataset.crop_size[0]}_{cfg.dataset.crop_size[1]}_{cfg.device_id}"
    subdir = perf_dir / today / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)

    # print("Simulate LR progression for first epochs")
    # print(f"scheduler.last_epoch: {scheduler.last_epoch}")
    # vals_last_lr = scheduler.get_last_lr()
    # print(f"scheduler.get_last_lr() returns:", [format(v, ".17g") for v in scheduler.get_last_lr()])
    # for epoch in range(0, 20):
    #     # lrs_before = [pg['lr'] for pg in optimizer.param_groups]
    #     # print(f"Epoch {epoch+1:2d} lr_before: {lrs_before}")

    #     # optimizer.step()
    #     scheduler.step()

    #     vals_last_lr = scheduler.get_last_lr()
    #     print(f"scheduler.last_epoch: {scheduler.last_epoch}")
    #     print("scheduler.get_last_lr() returns:", [format(v, ".17g") for v in vals_last_lr]) # -> step 후에 업데이트됨.

    #     # lrs_after = [pg['lr'] for pg in optimizer.param_groups]
    #     # print(f"Epoch {epoch+1:2d} lr_after : {lrs_after}\n")
    
    # return
    
    # train
    for epoch in range(cfg.trainer.epochs):
        epoch_start_time = datetime.now()

        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device, epoch, cfg.trainer.epochs)

        elapsed = datetime.now() - epoch_start_time
        minutes, seconds = divmod(elapsed.total_seconds(), 60)
        # print(f"train_loss: {train_loss}")
        # print(f"minutes: {minutes}")
        # print(f"seconds: {seconds}")

        print(f"Epoch [{epoch+1}/{cfg.trainer.epochs}]: Epoch Loss: {train_loss:.4f} | Elapsed Time: {int(minutes)}m {seconds:.2f}s")

        # -- DEBUG: print LR before stepping (this is the LR used during this epoch)
        # lrs_before = [pg['lr'] for pg in optimizer.param_groups]
        # print(f"[Epoch {epoch+1}] lr_before_step: {lrs_before}, train_loss: {train_loss:.6f}")

        scheduler.step()

        # -- DEBUG: print LR after stepping (this is the LR used during the next epoch)
        # lrs_after = [pg['lr'] for pg in optimizer.param_groups]
        # print(f"[Epoch {epoch+1}] lr_after_step: {lrs_after}")

        wandb.log({"train/loss": train_loss, "epoch": epoch})
        

        # evaluation
        # if epoch % cfg.trainer.eval_interval == 0:
        # # if (epoch + 1) % cfg.trainer.eval_interval == 0:
        #     performance = evaluate(
        #         model,
        #         valid_loader,
        #         device,
        #         data_root=Path(cfg.dataset.data_root),
        #         output_dir=preds_dir,
        #         wdb=wdb,
        #         epoch=epoch
        #     )
        #     if performance > best_performance:
        #         best_performance = performance
        #         print(f"[epoch: {epoch}], best_performance: {best_performance}")
        #         torch.save(model.state_dict(), subdir / "best_model.pth")
        
        # evaluation
        if epoch % cfg.trainer.eval_interval == 0:
        # if (epoch + 1) % cfg.trainer.eval_interval == 0:
            performance, performance_car = evaluate_only_car(
                model,
                valid_loader,
                device,
                data_root=Path(cfg.dataset.data_root),
                output_dir=preds_dir,
                wdb=wdb,
                epoch=epoch
            )
            if performance_car > best_performance:
                best_performance = performance_car
                print(f"[epoch: {epoch}], best_performance_car: {best_performance}")
                print(f"[epoch: {epoch}], performance: {performance}")
                torch.save(model.state_dict(), subdir / "best_model_car.pth")

if __name__ == "__main__":
    main()

