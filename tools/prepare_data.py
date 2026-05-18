"""
把 <SRC>/{images,annotations.json} 形态的 COCO 数据按比例划分为 train/val。
- 复制图像
- 按 image_id 划分 annotations.json 为 train.json / val.json
- 同时写出 train_img_list.txt / val_img_list.txt（绝对路径，每行一张）

用法：
    python tools/prepare_data.py \
        --src /data/raw/dior \
        --dst /data/dior_split \
        --val-ratio 0.1
"""
import argparse
import json
import random
import shutil
from pathlib import Path


def split_coco(coco, val_ratio, seed):
    rng = random.Random(seed)
    images = list(coco['images'])
    rng.shuffle(images)
    n_val = max(1, int(round(len(images) * val_ratio)))
    val_imgs = images[:n_val]
    train_imgs = images[n_val:]
    val_ids = {im['id'] for im in val_imgs}

    train_anns = [a for a in coco['annotations'] if a['image_id'] not in val_ids]
    val_anns = [a for a in coco['annotations'] if a['image_id'] in val_ids]

    base = dict(
        info=coco.get('info', {}),
        licenses=coco.get('licenses', []),
        categories=coco.get('categories', []),
    )
    train_coco = dict(base, images=train_imgs, annotations=train_anns)
    val_coco = dict(base, images=val_imgs, annotations=val_anns)
    return train_coco, val_coco


def write_split(split_name, coco_split, src_img_dir: Path, dst_root: Path):
    img_dst = dst_root / 'images' / split_name
    ann_dst = dst_root / 'annotations'
    img_dst.mkdir(parents=True, exist_ok=True)
    ann_dst.mkdir(parents=True, exist_ok=True)

    # 复制图像 + 写 list
    list_path = dst_root / 'config' / f'{split_name}_img_list.txt'
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(list_path, 'w', encoding='utf-8') as lf:
        for im in coco_split['images']:
            src = src_img_dir / im['file_name']
            dst = img_dst / Path(im['file_name']).name
            if not dst.exists() and src.exists():
                shutil.copy(src, dst)
            lf.write(str(dst.resolve()) + '\n')
    # COCO JSON
    ann_path = ann_dst / f'{split_name}.json'
    with open(ann_path, 'w', encoding='utf-8') as f:
        json.dump(coco_split, f, ensure_ascii=False)
    print(f'  [{split_name}] images={len(coco_split["images"])} '
          f'anns={len(coco_split["annotations"])} '
          f'-> {ann_path}', flush=True)
    return list_path, ann_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True,
                    help='原始数据根目录，应包含 images/ 和 annotations.json')
    ap.add_argument('--dst', required=True, help='输出根目录')
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    src_img = src / 'images'
    src_ann = src / 'annotations.json'
    assert src_img.is_dir(), f'找不到 {src_img}'
    assert src_ann.is_file(), f'找不到 {src_ann}'

    print(f'src = {src}')
    print(f'dst = {dst}')
    with open(src_ann, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    print(f'图像数: {len(coco["images"])}, 标注数: {len(coco["annotations"])}, '
          f'类别数: {len(coco.get("categories", []))}')

    train_coco, val_coco = split_coco(coco, args.val_ratio, args.seed)
    print(f'train: {len(train_coco["images"])}, val: {len(val_coco["images"])}')

    write_split('train', train_coco, src_img, dst)
    write_split('val',   val_coco,   src_img, dst)

    # mean/std 占位（按 ImageNet）
    cfg_dir = dst / 'config'
    cfg_dir.mkdir(exist_ok=True)
    with open(cfg_dir / 'mean_value.txt', 'w') as f:
        f.write('123.675\n116.28\n103.53\n')
    with open(cfg_dir / 'std_value.txt', 'w') as f:
        f.write('58.395\n57.12\n57.375\n')

    print('完成！')


if __name__ == '__main__':
    main()
