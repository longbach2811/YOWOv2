import os
import torch
import numpy as np
from scipy.io import loadmat

from dataset.custom_dataset import Custom_Dataset
from utils.box_ops import rescale_bboxes

from .cal_frame_mAP import evaluate_frameAP

class CustomDataset_Evaluator(object):
       def __init__(self,
                 data_root=None,
                 dataset='custom',
                 model_name='yowo',
                 metric='vmap',
                 img_size=224,
                 len_clip=1,
                 batch_size=1,
                 conf_thresh=0.01,
                 iou_thresh=0.5,
                 transform=None,
                 collate_fn=None,
                 gt_folder=None,
                 save_path=None):
        self.data_root = data_root
        self.dataset = dataset
        self.model_name = model_name
        self.img_size = img_size
        self.len_clip = len_clip
        self.batch_size = batch_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.collate_fn = collate_fn

        self.gt_folder = gt_folder
        self.save_path = save_path

        # dataset
        if metric == 'fmap':
            self.testset = Custom_Dataset(
                data_root=data_root,
                dataset=dataset,
                img_size=img_size,
                transform=transform,
                len_clip=len_clip,
                sampling_rate=1
            )
            self.num_classes = self.testset.num_classes

def evaluate_frame_map(self, model, epoch=1, show_pr_curve=False):
    print("Metric: Frame mAP")
    # dataloader
    self.testloader = torch.utils.data.DataLoader(
        dataset=self.testset, 
        batch_size=self.batch_size,
        shuffle=False,
        collate_fn=self.collate_fn, 
        num_workers=4,
        drop_last=False,
        pin_memory=True
        )
    epoch_size = len(self.testloader)

    # inference
    for iter_i, (batch_frame_id, batch_video_clip, batch_target) in enumerate(self.testloader):
        # to device
        batch_video_clip = batch_video_clip.to(model.device)

        with torch.no_grad():
            # inference
            batch_scores, batch_labels, batch_bboxes = model(batch_video_clip)

            # process batch
            for bi in range(len(batch_scores)):
                frame_id = batch_frame_id[bi]
                scores = batch_scores[bi]
                labels = batch_labels[bi]
                bboxes = batch_bboxes[bi]
                target = batch_target[bi]

                # rescale bbox
                orig_size = target['orig_size']
                bboxes = rescale_bboxes(bboxes, orig_size)

                if not os.path.exists('results'):
                    os.mkdir('results')

                if self.dataset == 'custom':
                    detection_path = os.path.join('results', 'custom_detections', self.model_name, 'detections_' + str(epoch), frame_id)
                    current_dir = os.path.join('results', 'custom_detections',  self.model_name, 'detections_' + str(epoch))
                    if not os.path.exists('results/custom_detections/'):
                        os.mkdir('results/custom_detections/')
                    if not os.path.exists('results/custom_detections/'+self.model_name):
                        os.mkdir('results/custom_detections/'+self.model_name)
                    if not os.path.exists(current_dir):
                        os.mkdir(current_dir)

                with open(detection_path, 'w+') as f_detect:
                    for score, label, bbox in zip(scores, labels, bboxes):
                        x = round(bbox[0])
                        y = round(bbox[1])
                        w = round(bbox[2])
                        h = round(bbox[3])
                        cls_id = int(label) + 1

                        f_detect.write(
                            str(cls_id) + ' ' + str(score) + ' ' \
                                + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')

            if iter_i % 100 == 0:
                log_info = "[%d / %d]" % (iter_i, epoch_size)
                print(log_info, flush=True)

    print('calculating Frame mAP ...')
    metric_list = evaluate_frameAP(self.gt_folder, current_dir, self.iou_thresh,
                            self.save_path, self.dataset, show_pr_curve)
    for metric in metric_list:
        print(metric)