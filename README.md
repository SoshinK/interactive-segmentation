## Interactive Segmentation

Initially cloned from https://github.com/saic-vul/ritm_interactive_segmentation

### Train
```
python3 train.py models/iter_mask/hrnet18s_cocolvis_itermask_3p.py --gpus=0 --workers=4 --exp-name=first-try
```


### Evaluation

Run evaluation script with visualizations and IoUs 

Example for baseline on Berkeley dataset:
```
python3 scripts/evaluate_model.py NoBRS --checkpoint=coco_lvis_h18_itermask --datasets=Berkeley --gpu=0  --iou-analysis --thresh=0.5 --save-ious --print-ious --vis-preds
```

Plot graphs with IoU/n_clicks curves

Example for baseline on Berkeley dataset:
```
python3 scripts/plot_ious_analysis.py --model-dirs /home/konstantin_soshin/Work/interactive-segmentation/experiments/evaluation_logs/others/coco_lvis_h18_itermask --dataset=Berkeley
```