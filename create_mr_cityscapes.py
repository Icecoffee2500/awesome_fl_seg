import os
import shutil
import sys
from PIL import Image
from datetime import datetime
from pathlib import Path

# def resize_and_copy(src_dir: Path, dst_dir: Path, res_w: int, res_h: int):
#     for root, dirs, files in os.walk(src_dir):
#         # send 폴더 무시
#         if 'send' in dirs:
#             dirs.remove('send')

#         # 상대 경로 계산 및 대상 디렉토리 생성
#         root_path = Path(root)
#         rel_path = root_path.relative_to(src_dir)
#         dst_root = dst_dir / rel_path
#         dst_root.mkdir(parents=True, exist_ok=True)

#         for file in files:
#             src_file = root_path / file
#             dst_file = dst_root / file

#             # gtFine 또는 leftImg8bit 하위의 PNG 이미지인 경우 리사이즈
#             if file.endswith('.png') and ('gtFine' in rel_path.parts or 'leftImg8bit' in rel_path.parts):
#                 with Image.open(src_file) as img:
#                     resized_img = img.resize((res_w, res_h), Image.BOX)  # BOX(AREA)으로 제대로 리사이즈
#                     resized_img.save(dst_file)
#             else:
#                 # 다른 파일은 그대로 복사
#                 shutil.copy2(src_file, dst_file)

import os
import shutil
import sys
from PIL import Image
from datetime import datetime
from pathlib import Path
import numpy as np

# optional: if opencv available
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

def resize_and_copy(src_dir: Path, dst_dir: Path, res_w: int, res_h: int):
    for root, dirs, files in os.walk(src_dir):
        if 'send' in dirs:
            dirs.remove('send')
        root_path = Path(root)
        rel_path = root_path.relative_to(src_dir)
        dst_root = dst_dir / rel_path
        dst_root.mkdir(parents=True, exist_ok=True)

        for file in files:
            src_file = root_path / file
            dst_file = dst_root / file

            if file.endswith('.png') and 'gtFine' in rel_path.parts:
                # GT 마스크: numpy + cv2.INTER_NEAREST (or PIL NEAREST fallback)
                with Image.open(src_file) as img:
                    arr = np.array(img)  # keep original dtype
                if _HAS_CV2:
                    resized = cv2.resize(arr, (res_w, res_h), interpolation=cv2.INTER_NEAREST)
                else:
                    # fallback: use PIL but ensure convert('L') to preserve single channel
                    with Image.fromarray(arr).convert('L') as im:
                        resized = np.array(im.resize((res_w, res_h), resample=Image.NEAREST))
                # save (uint8)
                Image.fromarray(resized).save(dst_file)
            elif file.endswith('.png') and 'leftImg8bit' in rel_path.parts:
                # 컬러 이미지 - area/box
                with Image.open(src_file) as img:
                    resized_img = img.resize((res_w, res_h), resample=Image.BOX)
                    resized_img.save(dst_file)
            else:
                shutil.copy2(src_file, dst_file)


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"복제 시작: {start_time.strftime('%Y년%m월%d일 %H시%M분%S초')}")
    if len(sys.argv) != 3:
        print("사용법: python create_mr_cityscapes.py <res_w> <res_h>")
        print("예시: python create_mr_cityscapes.py 1536 768")
        sys.exit(1)

    try:
        res_w = int(sys.argv[1])
        res_h = int(sys.argv[2])
    except ValueError:
        print("해상도는 정수로 입력해주세요.")
        sys.exit(1)

    ROOT = Path(__file__).parent
    print(f"ROOT: {ROOT}")

    src_dir = ROOT / "data" / "cityscapes"
    dst_dir = src_dir.parent / f"cityscapes_{res_w}x{res_h}"
    print(f"src_dir: {src_dir}")
    print(f"dst_dir: {dst_dir}")
    
    if dst_dir.exists():
        print(f"대상 폴더 {dst_dir}가 이미 존재합니다. 덮어쓰기를 피하려면 삭제 후 실행하세요.")
        sys.exit(1)

    print(f"복제 시작: {src_dir} -> {dst_dir} (해상도: {res_w}x{res_h})")
    resize_and_copy(src_dir, dst_dir, res_w, res_h)
    print(f"복제 완료: {datetime.now().strftime('%Y년%m월%d일 %H시%M분%S초')}")
    elapsed = datetime.now() - start_time
    minutes, seconds = divmod(elapsed.total_seconds(), 60)
    print(f"복제 및 리사이즈 완료! 소요 시간: {int(minutes)}m {seconds:.2f}s")