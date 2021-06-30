pip3 install -r requirements.txt
ln -s /group-volume/orc_srr/k.soshin/data/datasets/COCO+LVIS datasets/COCO+LVIS
ln -s /group-volume/orc_srr/k.soshin/pretrained_models/  pretrained_models
ln -s /group-volume/orc_srr/k.soshin/experiments/ experiments
python3 train.py models/iter_mask/hrnet18_cocolvis_itermask_3p.py --gpus=0,1,2,3 --workers=12 --exp-name=first-try
