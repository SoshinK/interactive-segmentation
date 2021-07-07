pip3 install -r requirements.txt
ln -s /group-volume/Domain-Adaptation/k.soshin/data datasets
ln -s /group-volume/Domain-Adaptation/k.soshin/pretrained_models/  pretrained_models
ln -s /group-volume/Domain-Adaptation/k.soshin/experiments/ experiments
# python3 train.py models/iter_mask/hrnet18_cocolvis_itermask_3p.py --gpus=0,1,2,3 --workers=12 --exp-name=first-try
#python3 train.py models/iter_mask/tos_hrnet18_cocolvis_itermask_3p.py --gpus=0,1,2,3 --workers=12 --exp-name=tos_hrnet_cls_aux_fusestream
# python3 train.py models/iter_mask/tos_hrnet18_cocolvis_itermask_3p.py --gpus=0,1,2 --workers=12 --batch-size=33 --exp-name=tos_hrnet_cls_edge_longer_with_edge_loss_bs25
