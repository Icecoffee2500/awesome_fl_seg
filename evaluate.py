from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval  # noqa
import glob

# valid_loader, model 정의되어 있다고 가정
# trainId: 0~18, labelId: 0~33
trainId_to_labelId = {
    0: 7,    # road
    1: 8,    # sidewalk
    2: 11,   # building
    3: 12,   # wall
    4: 13,   # fence
    5: 17,   # pole
    6: 19,   # traffic light
    7: 20,   # traffic sign
    8: 21,   # vegetation
    9: 22,   # terrain
    10: 23,  # sky
    11: 24,  # person
    12: 25,  # rider
    13: 26,  # car
    14: 27,  # truck
    15: 28,  # bus
    16: 31,  # train
    17: 32,  # motorcycle
    18: 33,  # bicycle
    255: 255 # ignore
}

def evaluate(model, valid_loader, device, data_root: Path, output_dir: Path, wdb=None, epoch: int=None) -> None:
    
    # Forward and save preds for cityscapes evaluation.
    _forward_and_save_preds(
        model,
        valid_loader,
        device,
        output_dir
    )

    # Evaluate through cityscapesscripts.
    metric = _evaluate(
        data_root=data_root,
        output_dir=output_dir
    )
    if wdb:
        wdb.log({"val/mIoU": metric['averageScoreClasses'], "epoch": epoch})

    # Print evaluation results.
    print("\n\t " + "-" * 33)
    print(f"\t| {'Category':<20} {'IoU':>10} |")
    print("\t " + "-" * 33)
    for cls, score in metric['classScores'].items():
        if np.isnan(score):   # NaN이면 출력하지 않음
            continue
        print(f"\t| {cls:<20} {score:>10.4f} |")
    print("\t " + "-" * 33)
    print(f"\t| {'Average IoU':<20} {metric['averageScoreClasses']:>10.4f} |")
    print("\t " + "-" * 33 + "\n")

    return metric['averageScoreClasses']
        
@torch.no_grad()
def _forward_and_save_preds(model, valid_loader, device, output_dir):
    """Forward and save preds for cityscapes evaluation.

    valid_loader에서 데이터 꺼내서 preds 생성한 후에
    gt랑 shape 다르면 맞춰주고
    output_dir에 저장

    """
    model.eval()

    for inputs, masks, img_infos in valid_loader:
        inputs = inputs.to(device)
        masks = masks.to(device)

        outputs = model(inputs)
        # outputs가 [B, C, H, W]일 경우 argmax
        if outputs.dim() == 4:
            preds = outputs.argmax(1)  # [B, H, W]
        else:
            preds = outputs

        # masks와 동일한 크기로 맞추기
        if preds.shape[-2:] != masks.shape[-2:]:
            preds = F.interpolate(
                preds.unsqueeze(1).float(),
                size=masks.shape[-2:],
                mode="nearest"
            ).squeeze(1).long()

        # 각 이미지별 PNG 저장
        for idx in range(preds.shape[0]): # batch 안에 있는 sample의 index
            pred_np = preds[idx].cpu().numpy().astype(np.uint8)
            pred_np_labelId = np.vectorize(trainId_to_labelId.get)(pred_np).astype(np.uint8)

            # img_infos[b]에는 원래 Cityscapes 이미지 이름과 폴더 구조 정보 필요
            # 예: "aachen_000000_000019.png"
            img_name = img_infos['file_name'][idx]
            city_name = img_infos['city'][idx]
            save_path = output_dir / city_name
            save_path.mkdir(parents=True, exist_ok=True)
            # Image.fromarray(pred_np).save(save_path / img_name)
            Image.fromarray(pred_np_labelId).save(save_path / img_name)

# 여러 resolution의 data를 평가하기 위해서는
# data_root를 resolution 별로 만들어놔야 한다.
# dataset.test_scale에 해당하는 data_root를 넘겨주면 된다.
def _evaluate(data_root: Path, output_dir: Path) -> dict:
    # 모든 prediction PNG 저장 후 평가
    gt_dir = data_root / "gtFine" / "val" # "/path/to/gtFine/val"  # GT 폴더 경로
    gt_dir = gt_dir.expanduser()
    print(f"gt_dir: {gt_dir}")
    gt_imgs = sorted(glob.glob(f"{gt_dir}/**/*_gtFine_labelIds.png", recursive=True))
    pred_imgs = sorted(glob.glob(f"{output_dir}/**/*.png", recursive=True))

    assert len(gt_imgs) == len(pred_imgs), "GT와 prediction 이미지 개수 불일치"

    # set cityscapesscripts args
    eval_results = dict()
    CSEval.args.evalInstLevelScore = True
    CSEval.args.predictionPath = output_dir
    CSEval.args.evalPixelAccuracy = True
    CSEval.args.JSONOutput = False
    CSEval.args.quiet = True

    # evaluate through cityscapesscripts
    metric = dict()
    eval_results.update(
        CSEval.evaluateImgLists(pred_imgs, gt_imgs, CSEval.args))
    # print(eval_results.keys())
    metric['classScores'] = eval_results['classScores']
    metric['averageScoreClasses'] = eval_results['averageScoreClasses']

    return metric