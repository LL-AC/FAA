import torch.utils.data as data
import pandas as pd
import glob
import os
from PIL import Image
import numpy as np
import torch
from scipy import stats
import json
import cv2
from torchvision import transforms
from tqdm import tqdm


class Tmed2(data.Dataset):
    def __init__(self, path, transform=None, mode='train'):
        super(Tmed2, self).__init__()
        self.cls_mapping = {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS':2, 'severe_AS': 2}
        self.view_mapping = {'PLAX':0, 'PSAX': 1, 'A4CorA2CorOther': 2, 'A4C':2,'A2C':2}
        self.transform = transform
        self.mode = mode
        self.dataset = pd.read_csv(path)
        self.dataset = self.dataset[(self.dataset['diagnosis_classifier_split'] == mode) & 
                                    (self.dataset['diagnosis_label'] != 'Not_Provided')]
        self.dataset["case_id"] = self.dataset ["query_key"].apply(lambda x: x.split("_")[0])
        self.img_path = 'TMED2/approved_users_only/view_and_diagnosis_labeled_set/labeled'

        # 3. 得到【不重复的 caseid 列表】
        self.case_list = self.dataset["case_id"].unique().tolist()

    def __getitem__(self, idx):
        # 1. 获取当前病例ID
        case_id = self.case_list[idx]

        # 2. 从df筛选出属于这个case的所有行（核心！）
        case_df = self.dataset[self.dataset['case_id'] == case_id]

        # 3. 取出该病例的 所有图片路径、视角标签、病例级标签
        img_paths = case_df['query_key'].tolist()       # 图片路径列表
        view_labels = case_df['view_label'].tolist()    # 视角列表
        case_label = self.cls_mapping[case_df['diagnosis_label'].iloc[0]] # 病例级标签（取一个就行）

        # 4. 读取图片 + 转换视角标签
        imgs, viewlabel = self.read_imgs(img_paths, view_labels)
        
        # 转tensor
        label = torch.tensor(case_label, dtype=torch.long)
        viewlabel = torch.tensor(viewlabel, dtype=torch.long)

        return {
            'img': imgs, 
            'label': label, 
            'case_id': case_id, 
            'view_label': viewlabel, 
            'paths': img_paths
        }

    # ------------------------------
    # 修正后的 read_imgs（极简）
    # ------------------------------
    def read_imgs(self, paths, view_labels):
        try:
            imgs = []
            view_ids = []

            # 遍历这个病例的所有图片
            for path, view_str in zip(paths, view_labels):
                # 读图

                img = Image.open(os.path.join(self.img_path, path)).convert('RGB')
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                # 增强
                if self.transform is not None:
                    img = self.transform(image=img)['image']
                img = transforms.ToTensor()(img)

                imgs.append(img)
                # view 映射成数字
                view_ids.append(self.view_mapping[view_str])

           # print(len(imgs), imgs[0])
            # 堆叠成 tensor
            imgs = torch.stack(imgs, dim=0)  # [N, 3, 224, 224]

        except Exception as e:
            raise Exception(f'读图失败: {e}')
        
        return imgs, view_ids
    def __len__(self):
        return len(self.case_list)