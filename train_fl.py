from pathlib import Path
import copy
import torch
from segmentation.datasets.cityscapes import CityscapesDataset
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
# import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Segformer
from utils import save_data, print_keys, set_seed
from evaluate import evaluate
from segmentation.optimizer import build_optimizer
from segmentation.scheduler import build_epoch_scheduler
import wandb
from datetime import datetime
import torch.nn.functional as F
from functools import partial
from PIL import Image
import numpy as np

ROOT = Path.cwd()

# Federated averaging: FedAvg
def fed_avg(weights: list, num_samples: list = None):
    if num_samples is None:
        num_samples = [1 for _ in range(len(weights))]
    # print(f"num_samples = {num_samples}")
    w_avg = {k: torch.zeros_like(v) for k, v in weights[0].items()}  # OrderedDict에서 각 텐서를 사용하여 초기화
    
    for k in w_avg.keys():
        for i in range(len(weights)):
            w_avg[k] += weights[i][k].detach() * num_samples[i]
        w_avg[k] = torch.div(w_avg[k], sum(num_samples))
    return w_avg

def aggregate(weights, num_samples=None):
    aggregation = partial(fed_avg, num_samples=num_samples)
    
    w_glob_client = aggregation(weights) # 각 client에서 update된 weight를 받아서 FedAvg로 합쳐줌.
    print(f"\n>>> Fed Server: Weights are aggregated.\n")
    
    return w_glob_client

def train_one_epoch_fl(model, optimizer, criterion, train_loader, device, curr_epoch, max_epochs, client_idx, target_resolutions):
    model.train()
    epoch_loss = 0.0
    iter_start_time = datetime.now()

    # client별 target resolution 설정
    # target_resolutions = {
    #     0: (1024, 1024),
    #     1: (768, 768),
    #     2: (512, 512)
    #     # 0: (1024, 1024),
    #     # 1: (1024, 1024),
    #     # 2: (1024, 1024)
    # }
    target_h, target_w = target_resolutions.get(client_idx, (1024, 1024))

    for idx, (inputs, masks, img_infos) in enumerate(train_loader):
        # client별 resolution으로 resize
        if (inputs.shape[2], inputs.shape[3]) != (target_h, target_w):
            inputs = F.interpolate(inputs, size=(target_h, target_w), mode='bilinear', align_corners=False)
            masks = F.interpolate(masks.float(), size=(target_h, target_w), mode='nearest').long()
        # print(inputs.shape)
        # print(masks.shape)
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
            # loss 높게 나올 때 어느 class인지 확인하기
            print(f"Client {client_idx+1} | Epoch [{curr_epoch+1}/{max_epochs}][{idx}/{len(train_loader)}]: "
                f"Current Loss: {loss.item():.4f} | Elapsed Time: {int(minutes)}m {seconds:.2f}s")

            # print(f"Epoch [{curr_epoch+1}/{max_epochs}][{idx}/{len(train_loader)}]: Loss: {loss.item():.4f} | Time: {datetime.now() - epoch_start_time:.2f}s")
            iter_start_time = datetime.now()
    
    return epoch_loss / len(train_loader)

@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(cfg:DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # config = OmegaConf.to_container(cfg, resolve=True) # dict로 변환 (resolve=True: 참조 해결)
    
    now = datetime.now()
    today = now.strftime("%m%d-%H%M")
    
    name = "train_fl_"
    name += f"{cfg.fl.target_resolutions}"
    name += f"{cfg.dataset.name}_gpu-{cfg.device_id}"
    name += f"_{today}"
    wdb = wandb
    wdb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        project="Segmentation training (FL)",
        name = name,
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

    # train_dataset 길이
    dataset_len = len(train_dataset)

    # 클라이언트 수
    # num_clients = 3
    num_clients = 6

    # 각 클라이언트 dataset 길이 계산
    split_len = dataset_len // num_clients
    client_dataset_lengths = [split_len] * (num_clients - 1)
    client_dataset_lengths.append(dataset_len - sum(client_dataset_lengths))  # 나머지는 마지막에

    # dataset 나누기
    client_datasets = torch.utils.data.random_split(train_dataset, client_dataset_lengths)
    
    val_dataset = CityscapesDataset(
        root=cfg.dataset.data_root,
        split="val",
        mode=cfg.dataset.mode,
        target_type=cfg.dataset.target_type,
        pipeline_cfg=cfg.dataset.test_pipeline,
    )

    # 데이터로더 설정
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=cfg.dataset.train_dataloader.batch_size,
    #     shuffle=cfg.dataset.train_dataloader.sampler.shuffle,
    #     num_workers=cfg.dataset.train_dataloader.num_workers,
    #     pin_memory=True,
    # )
    train_loaders = []
    for i in range(num_clients):
        train_loaders.append(
            DataLoader(
                client_datasets[i],
                batch_size=cfg.dataset.train_dataloader.batch_size,
                shuffle=cfg.dataset.train_dataloader.sampler.shuffle,
                num_workers=cfg.dataset.train_dataloader.num_workers,
                pin_memory=True,
            )
    )
    
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
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # checkpoint = "smp-hub/segformer-b0-1024x1024-city-160k"
    # model = smp.from_pretrained(checkpoint).to(device)
    
    global_model = Segformer(
        encoder_name="mit_b0",            # MiT-B0
        encoder_weights="imagenet",       # ImageNet pretrained
        decoder_channels=256,             # decoder 내부 채널
        in_channels=3,
        classes=19,                       # Cityscapes 클래스 수
        encoder_output_stride=32          # stage 0~3 출력 stride 설정 (smp 기본값)
    )
    global_model.to(device)

    # optimizer, scheduler, criterion 설정
    # global_optimizer = build_optimizer(global_model, cfg.optimizer)
    # scheduler = build_epoch_scheduler(global_optimizer, cfg.scheduler)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    model_list = []
    optimizer_list = []
    scheduler_list = []
    for i in range(3):
        model = copy.deepcopy(global_model)
        model.to(device)
        model_list.append(model)
        optimizer = build_optimizer(model, cfg.optimizer)
        optimizer_list.append(optimizer)
        scheduler_list.append(build_epoch_scheduler(optimizer, cfg.scheduler))

    # evaluation
    # preds_dir = ROOT / "predictions"  # 예측 파일 저장 경로
    preds_dir = ROOT / f"predictions_{cfg.device_id}"  # 예측 파일 저장 경로
    preds_dir.mkdir(parents=True, exist_ok=True)

    best_performance = 0.0
    perf_dir = ROOT / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)

    # subdir_name = f"fl_{cfg.device_id}"
    
    # subdir = perf_dir / today / subdir_name
    subdir = perf_dir / today
    subdir.mkdir(parents=True, exist_ok=True)

    save_name = "fl_"
    for res in cfg.fl.target_resolutions.values():
        save_name += f"{res[0]}x{res[1]}_"
    save_name += f"gpu{cfg.device_id}_best_model.pth"

    # train
    loss_list = []
    client_weights = []
    for epoch in range(cfg.trainer.epochs):
        epoch_start_time = datetime.now()
        client_weights = []

        for i, (model, optimizer, train_loader) in enumerate(zip(model_list, optimizer_list, train_loaders)):
            train_loss = train_one_epoch_fl(
                model, optimizer, criterion, train_loader, device, epoch, cfg.trainer.epochs, client_idx=i, target_resolutions=cfg.fl.target_resolutions
            )
            loss_list.append(train_loss)
            client_weights.append(model.state_dict())
        
        # aggregate weights
        print(f">>> load Fed-Averaged weight to the proxy client model ...")
        # w_glob_client = aggregate(client_weights)
        w_glob_client = aggregate(client_weights, client_dataset_lengths)

        # Braadcast weight to each clients
        print(f">>> load Fed-Averaged weight to the each client model ...")
        for model in model_list:
            model.load_state_dict(w_glob_client)
        
        global_model.load_state_dict(w_glob_client)

        elapsed = datetime.now() - epoch_start_time
        minutes, seconds = divmod(elapsed.total_seconds(), 60)

        print(f"Epoch [{epoch + 1}/{cfg.trainer.epochs}]: Epoch Loss: {train_loss:.4f} | Elapsed Time: {int(minutes)}m {seconds:.2f}s")
        for scheduler in scheduler_list:
            scheduler.step()
        wandb.log({"train/loss": train_loss, "epoch": epoch})

        # evaluation
        print(f"validation dataloader length: {len(valid_loader)}")
        if epoch % cfg.trainer.eval_interval == 0:
            performance =evaluate(
                global_model,
                valid_loader,
                device,
                data_root=Path(cfg.dataset.data_root),
                output_dir=preds_dir,
                wdb=wdb,
                epoch=epoch
            )
            if performance > best_performance:
                best_performance = performance
                # #TODO: checkpoint 저장 경로 폴더 날짜별로 만들어야 함.
                # torch.save(model.state_dict(), perf_dir / f"best_model_fl.pth")
                best_performance = performance
                # torch.save(global_model.state_dict(), subdir / "best_model_fl.pth")
                torch.save(global_model.state_dict(), subdir / save_name)
            
            print(f"[epoch: {epoch}], best_performance: {best_performance}")

if __name__ == "__main__":
    main()

