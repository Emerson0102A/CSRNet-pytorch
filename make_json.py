import json, glob, os

root = "/root/autodl-tmp/Shanghai"  # 数据集根目录
train_imgs = sorted(glob.glob(os.path.join(root, "part_A_final/train_data/images", "*.jpg")))
val_imgs   = sorted(glob.glob(os.path.join(root, "part_A_final/test_data/images", "*.jpg")))

with open("train.json", "w") as f:
    json.dump(train_imgs, f)

with open("val.json", "w") as f:
    json.dump(val_imgs, f)
