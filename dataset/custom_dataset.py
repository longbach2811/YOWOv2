import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 dataset = 'custom',
                 img_size = 224,
                 transform=None,
                 is_train = False,
                 len_clip=16,
                 split_ratio=0.8,
                 sampling_rate=1):
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform

        self.img_size = img_size
        self.len_clip = len_clip
        self.is_train = is_train
        self.sampling_rate = sampling_rate
        
        
        self.classes = self.load_classes(os.path.join(root_dir, 'classes.txt'))
        self.samples = self.load_samples()

        self.train_size = int(split_ratio * len(self.samples))
        self.train_dataset, self.test_dataset = self.samples[:self.train_size], self.samples[self.train_size:]

    def __len__(self):
        if self.is_train:
            return len(self.train_dataset)
        else:
            return len(self.test_dataset)
        

    def __getitem__(self, idx):
        frame_idx, video_clip, target = self.pull_item(idx)

        return frame_idx, video_clip, target
    
    def pull_item(self, idx):
        if self.is_train:
            img_path, label_path = self.train_dataset[idx]
            d = random.randint(1, 2)
            max_num = len(self.train_dataset)
        else:
            img_path, label_path = self.test_dataset[idx]
            d = self.sampling_rate
            max_num = len(self.train_dataset)
        img_split = label_path.split("/")
        img_id = img_split[-1].split("_")[-1].split(".")[0]
        video_clip = []
        for i in reversed(range(self.len_clip)):
            img_id_temp = int(img_id) - i * d
            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > max_num:
                img_id_temp = max_num
            
            frame = Image.open(img_path).convert('RGB')

            # if self.transform:
            #     frame = self.transform(frame)
            ow, oh = frame.width, frame.height
            # ow, oh = frame.size(2), frame.size(1)

            video_clip.append(frame)

            frame_id = f"{img_split[6]}_{img_split[7]}_{img_split[8]}"
        
        if os.path.getsize(label_path):
            target = np.loadtxt(label_path)
        else:
            target = None 

        label = target[..., :1]
        # print(label)
        boxes = target[..., 1:]
        # print(boxes)
        target = np.concatenate([boxes, label], axis=-1).reshape(-1, 5)  
        # print(target)
        
        # transform
        video_clip, target = self.transform(video_clip, target)
        # List [T, 3, H, W] -> [3, T, H, W]
        video_clip = torch.stack(video_clip, dim=1)

        # reformat target
        target = {
            'boxes': target[:, :4],      # [N, 4]
            'labels': target[:, -1] - 1,    # [N,]
            'orig_size': [ow, oh],
            'video_idx':frame_id
            }
        print(target)

        return frame_id, video_clip, target     
        
        

    def load_classes(self, classes_file):
        with open(classes_file, 'r') as file: 
            classes = [line.strip() for line in file.readlines()]
            self.num_classes = len(classes)
            return classes
        

    def load_samples(self):
        samples = []

        for class_folder in os.listdir(self.root_dir):
            class_folder_path = os.path.join(self.root_dir, class_folder)
            # print("class_folder_path: ", class_folder_path)
            if os.path.isdir(class_folder_path):
                data_folder_path = os.path.join(class_folder_path, "labels")
                # print("data folder path:", data_folder_path)
                for file in os.listdir(data_folder_path):
                    if file.endswith(".txt"):
                        frame_num = file.split("_")[-1].split(".")[0]
                        label_path = os.path.join(data_folder_path, file)
                        img_path = os.path.join(class_folder_path, "images", f"frame_{frame_num}.jpg")
                        # print(img_path, label_path)
                        if os.path.exists(label_path) and os.path.exists(img_path):
                            samples.append((img_path, label_path))
        # print(len(samples))               
        return samples


if __name__ == "__main__":
    import cv2
    from transforms import Augmentation, BaseTransform

    root_dir = '/home/longbach/Desktop/motion-det-dataset/processed_data'
    trans_config = {
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5
    }

    train_transform = Augmentation(
        img_size=224,
        jitter=trans_config['jitter'],
        saturation=trans_config['saturation'],
        exposure=trans_config['exposure']
        )
    val_transform = BaseTransform(img_size=224)

    train_dataset = CustomDataset(root_dir=root_dir,
                                  img_size=224,
                                  transform=train_transform,
                                  is_train=True,
                                  len_clip=16,
                                  sampling_rate=1)
    print(len(train_dataset))
    for i in range(len(train_dataset)):
        frame_id, video_clip, target = train_dataset[i]
        key_frame = video_clip[:, -1, :, :]

        # to numpy
        key_frame = key_frame.permute(1, 2, 0).numpy()
        key_frame = key_frame.astype(np.uint8)

        # to BGR
        key_frame = key_frame[..., (2, 1, 0)]
        H, W, C = key_frame.shape

        key_frame = key_frame.copy()
        bboxes = target['boxes']
        labels = target['labels']

        for box, cls_id in zip(bboxes, labels):
            x1, y1, x2, y2 = box
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            key_frame = cv2.rectangle(key_frame, (x1, y1), (x2, y2), (255, 0, 0))

        # cv2 show
        cv2.imshow('key frame', key_frame)
        cv2.waitKey(0)
