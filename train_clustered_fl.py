from pathlib import Path
import copy
import torch
from segmentation.datasets.cityscapes import CityscapesDataset
from segmentation.datasets.cityscapes_one_class import CityscapesDatasetOneClass
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
from sklearn.cluster import KMeans
from collections import Counter

ROOT = Path.cwd()

# Federated averaging: FedAvg
def fed_avg(weights: list, num_samples: list = None):
    if num_samples is None:
        num_samples = [1 for _ in range(len(weights))]
    # print(f"num_samples = {num_samples}")
    w_avg = {k: torch.zeros_like(v) for k, v in weights[0].items()}  # OrderedDict에서 각 텐서를 사용하여 초기화
    
    for k in w_avg.keys():
        # for i in range(len(weights)):
        for i in range(len(num_samples)):
            w_avg[k] += weights[i][k].detach() * num_samples[i]
        w_avg[k] = torch.div(w_avg[k], sum(num_samples))
    return w_avg

def aggregate(weights, num_samples=None):
    aggregation = partial(fed_avg, num_samples=num_samples)
    
    w_glob_client = aggregation(weights) # 각 client에서 update된 weight를 받아서 FedAvg로 합쳐줌.
    print(f"\n>>> Fed Server: Weights are aggregated.\n")
    
    return w_glob_client

def train_one_epoch_fl(model, optimizer, criterion, train_loader, device, curr_epoch, max_epochs, client_idx):
    model.train()
    epoch_loss = 0.0
    iter_start_time = datetime.now()

    for idx, (inputs, masks, img_infos) in enumerate(train_loader):
        inputs = inputs.to(device)
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
            print(f"Client {client_idx + 1} | Epoch [{curr_epoch + 1}/{max_epochs}][{idx}/{len(train_loader)}]: "
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

    device = f"cuda:{cfg.device_id}" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # label_id = 31 # train class label_id
    label_id = 26 # car class label_id
    train_id = 13 # car class train_id

    # 데이터셋 로드
    # train_dataset = CityscapesDataset(
    #     root=cfg.dataset.data_root,
    #     split="train",
    #     mode=cfg.dataset.mode,
    #     target_type=cfg.dataset.target_type,
    #     pipeline_cfg=cfg.dataset.train_pipeline,
    # )
    train_dataset = CityscapesDatasetOneClass(
        root=cfg.dataset.data_root,
        split="train",
        mode=cfg.dataset.mode,
        target_type=cfg.dataset.target_type,
        pipeline_cfg=cfg.dataset.train_pipeline,
        train_id=train_id,
    )

    # train_dataset에서 특정 class만 뽑아서 train_dataset_with_one_class 생성 ---------
    print(f"len(train_dataset): {len(train_dataset)}")
    
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
    # print(f"indieces with class: {indices_with_class}")

    # 방법 A: dataset에 인덱스로 접근 가능한 경우 (권장)
    scales = []
    dataset_indices = []   # dataset index와 1:1 매핑을 유지
    for idx in range(len(train_dataset)):
        _, _, img_info = train_dataset[idx]
        scale = img_info['scale']
        scales.append(scale)
        dataset_indices.append(idx)
    
    scales = np.array(scales)
    print(f"scales: {scales}")
    scales_np = scales.reshape(-1, 1)   # shape (N,1)
    
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(scales_np)
    print(f"kmeans: {kmeans}")

    # choose labels = labels_kmeans if available else labels_quantile
    labels = kmeans.labels_   # 0,1,2
    print(f"labels: {labels}")

    # centroids
    centroids = kmeans.cluster_centers_.flatten()   # shape (k,)
    # centroids 오름차순 정렬 -> order는 original cluster index들을 작은값부터 담고 있음
    order = np.argsort(centroids)                   # e.g. array([2,0,1]) 의미: cluster2가 가장 작음
    # original cluster idx -> rank(0=small,1=mid,2=large)
    rank_of_cluster = {int(orig_idx): int(rank) for rank, orig_idx in enumerate(order)}

    # labels remap: original label -> rank label (0=small,1=mid,2=large)
    labels_sorted = np.array([rank_of_cluster[int(l)] for l in labels])

    # build group indices
    group_indices = {0: [], 1: [], 2: []}
    for i, lab in enumerate(labels_sorted):
        ds_idx = dataset_indices[i]
        group_indices[int(lab)].append(ds_idx)

    # 확인
    counts = Counter(labels_sorted)
    print(f"After remapping (KMeans -> small / mid / large): {counts}")
    print(f"Sorted centroids (small -> large): {centroids[order]}")

    # # 그룹별 인덱스 (dataset 인덱스)
    # group_indices = {0: [], 1: [], 2: []}
    # for i, lab in enumerate(labels):
    #     ds_idx = dataset_indices[i]   # dataset index corresponding to the i-th collected scale
    #     group_indices[int(lab)].append(ds_idx)

    # 확인
    for g in [0, 1, 2]:
        print(f"group {g}: {len(group_indices[g])} samples")
    
    # Subset 생성
    client_dataset_lengths = []
    client_datasets = {}
    for g in [0, 1, 2]:
        inds = group_indices[g]
        client_datasets[g] = torch.utils.data.Subset(train_dataset, inds)  # train_dataset은 원본 전체 dataset 객체
        client_dataset_lengths.append(len(client_datasets[g]))

    print(f"client_dataset_lengths: {client_dataset_lengths}")

    # train_loader_temp = DataLoader(
    #     train_dataset_with_one_class,
    #     batch_size=cfg.dataset.train_dataloader.batch_size,
    #     shuffle=cfg.dataset.train_dataloader.sampler.shuffle,
    #     num_workers=cfg.dataset.train_dataloader.num_workers,
    #     pin_memory=True,
    # )
    # for idx, (inputs, masks, img_infos) in enumerate(train_loader_temp):
    #     # print(f"inputs.shape: {inputs.shape}")
    #     # print(f"masks.shape: {masks.shape}")
    #     print(f"img_infos['area']: {img_infos['area']}")
    #     print(f"img_infos['scale']: {img_infos['scale']}")
    #     if idx > 10:
    #         break
    #     # break


    # return
    # -------------------------------------------------------------------------

    # # train_dataset 길이
    # dataset_len = len(train_dataset)

    # # 클라이언트 수
    # num_clients = 3

    # # 각 클라이언트 dataset 길이 계산
    # split_len = dataset_len // num_clients
    # lengths = [split_len] * (num_clients - 1)
    # lengths.append(dataset_len - sum(lengths))  # 나머지는 마지막에

    # # dataset 나누기
    # client_datasets = torch.utils.data.random_split(train_dataset, lengths)

    # 사용 예시
    # client1_dataset = client_datasets[0]
    # client2_dataset = client_datasets[1]
    # client3_dataset = client_datasets[2]
    
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
    # train_loaders = []
    # for i in range(num_clients):
    #     train_loaders.append(
    #         DataLoader(
    #             client_datasets[i],
    #             batch_size=cfg.dataset.train_dataloader.batch_size,
    #             shuffle=cfg.dataset.train_dataloader.sampler.shuffle,
    #             num_workers=cfg.dataset.train_dataloader.num_workers,
    #             pin_memory=True,
    #         )
    # )
    client_train_loaders = []
    for g in [0, 1, 2]:
        client_train_loaders.append(
            DataLoader(
                client_datasets[g],
                batch_size=cfg.dataset.train_dataloader.batch_size,
                shuffle=cfg.dataset.train_dataloader.sampler.shuffle,
                num_workers=cfg.dataset.train_dataloader.num_workers,
                pin_memory=True,
            )
        )
    # for i in range(client_datasets.keys()):
    #     train_loaders.append(
    #         DataLoader(
    #             client_datasets[i],
    #             batch_size=cfg.dataset.train_dataloader.batch_size,
    #             shuffle=cfg.dataset.train_dataloader.sampler.shuffle,
    #             num_workers=cfg.dataset.train_dataloader.num_workers,
    #             pin_memory=True,
    #         )
    # )
    
    valid_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.val_dataloader.batch_size,
        shuffle=cfg.dataset.val_dataloader.sampler.shuffle,
        num_workers=cfg.dataset.val_dataloader.num_workers,
        pin_memory=True,
    )

    # 모델 로드
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

        for i, (model, optimizer, train_loader) in enumerate(zip(model_list, optimizer_list, client_train_loaders)):
            train_loss = train_one_epoch_fl(
                model, optimizer, criterion, train_loader, device, epoch, cfg.trainer.epochs, client_idx=i, target_resolutions=cfg.fl.target_resolutions
            )
            loss_list.append(train_loss)
            client_weights.append(model.state_dict())
        
        # aggregate weights
        print(f">>> load Fed-Averaged weight to the proxy client model ...")
        # w_glob_client = aggregate(client_weights)
        w_glob_client = aggregate(client_weights, client_dataset_lengths)

        # Broadcast weight to each clients
        print(f">>> load Fed-Averaged weight to the each client model ...")
        for model in model_list:
            model.load_state_dict(w_glob_client)
        
        global_model.load_state_dict(w_glob_client)

        elapsed = datetime.now() - epoch_start_time
        minutes, seconds = divmod(elapsed.total_seconds(), 60)

        print(f"Epoch [{epoch+1}/{cfg.trainer.epochs}]: Epoch Loss: {train_loss:.4f} | Elapsed Time: {int(minutes)}m {seconds:.2f}s")
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

