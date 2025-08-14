#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.num_classes = 1
        self.data_dir = "/data/code/image-detector/train_yolox/YOLOX/datasets/circle_dataset"
        self.train_ann = "train_labels.json"
        self.val_ann = "val_labels.json"
        self.test_ann = "test_labels.json"
        
        # Memory optimization settings
        self.data_num_workers = 4  # Minimal workers to save memory
        self.batch_size = 2  # Increased batch size for better training performance
        self.input_size = (640, 640)  # Standard YOLOX size for better accuracy
        self.test_size = (640, 640)  # Standard YOLOX size for better accuracy
        self.multiscale_range = 0  # Disable multiscale training to save memory
        self.cache = True
        
        # Training settings
        self.max_epoch = 50  # Restored to normal training epochs
        self.warmup_epochs = 10 # Restored to normal warmup epochs
        self.basic_lr_per_img = 0.01 / 64.0  # Reduced learning rate for stability
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.weight_decay = 5e-4
        self.momentum = 0.937
        self.print_interval = 4  # Print only at the end of each epoch (4 iterations)
        self.eval_interval = 10
        self.save_interval = 10
        
        # Augmentation settings - Reduced for initial training stability
        self.degrees = 5.0  # Reduced rotation
        self.translate = 0.05  # Reduced translation
        self.scale = (0.8, 1.2)  # Reduced scaling range
        self.mscale = (0.9, 1.1)  # Reduced multiscale range
        self.shear = 1.0  # Reduced shear
        self.perspective = 0.0
        self.flipud = 0.0
        self.flip = 0.5
        self.mosaic = 0.0 # Reduced mosaic probability
        self.mixup = 0.0  # Reduced mixup probability
        
        # Evaluation settings
        self.test_conf = 0.05  # Increased confidence threshold for evaluation
        self.nmsthre = 0.65
        self.iou_thres = 0.5  # IoU threshold for evaluation (0.5 is standard)
        self.test_size = (640, 640)  # Ensure test size is set for evaluation
        
        # Export settings
        self.export_input_names = ["input"]
        self.export_output_names = ["output"]
        self.export_dynamic = True
        
        # Additional evaluation parameters
        self.eval_interval = 10
        self.save_interval = 10
        self.eval_mode = "COCO"  # Use COCO evaluation mode
        self.max_labels = 50  # Maximum number of labels per image
        
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """Override get_data_loader method as required by YOLOX documentation"""
        from yolox.data import COCODataset, TrainTransform, MosaicDetection, YoloBatchSampler, DataLoader, InfiniteSampler, worker_init_reset_seed
        
        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train/images",
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=self.max_labels, flip_prob=self.flip, hsv_prob=0.015),
            cache=cache_img,
            cache_type="ram",
        )
        
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=self.max_labels, flip_prob=self.flip, hsv_prob=0.015),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.scale,  # Fixed: use mosaic_scale instead of scale
            shear=self.shear,
            enable_mixup=self.mixup,
        )
        
        if is_distributed:
            import torch.distributed as dist
            batch_size = batch_size // dist.get_world_size()
        
        sampler = InfiniteSampler(len(dataset), seed=self.seed if self.seed else 0)
        
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        
        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        
        return DataLoader(dataset, **dataloader_kwargs)
    
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Override get_eval_loader method as required by YOLOX documentation"""
        from yolox.data import COCODataset, ValTransform
        import torch
        from torch.utils.data import DataLoader
        
        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val/images",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
        
        if is_distributed:
            batch_size = batch_size // torch.distributed.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "sampler": sampler,
            "batch_size": batch_size,
            "pin_memory": True,
            "drop_last": False,
        }
        
        return DataLoader(valdataset, **dataloader_kwargs)
    
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Override get_evaluator method as required by YOLOX documentation"""
        from yolox.evaluators import COCOEvaluator
        
        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev, legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
    
    def verify_dataset_config(self):
        """Verify dataset configuration and print debug info"""
        print(f"Dataset configuration:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Train annotations: {self.train_ann}")
        print(f"  Val annotations: {self.val_ann}")
        print(f"  Test annotations: {self.test_ann}")
        print(f"  IoU threshold: {getattr(self, 'iou_thres', 'Not set')}")
        print(f"  Test confidence: {self.test_conf}")
        print(f"  NMS threshold: {self.nmsthre}")
        
        # Verify dataset files exist
        import os
        train_ann_path = os.path.join(self.data_dir, self.train_ann)
        val_ann_path = os.path.join(self.data_dir, self.val_ann)
        
        print(f"\nDataset file verification:")
        print(f"  Train annotations exist: {os.path.exists(train_ann_path)}")
        print(f"  Val annotations exist: {os.path.exists(val_ann_path)}")
        
        if os.path.exists(train_ann_path):
            import json
            with open(train_ann_path, 'r') as f:
                train_data = json.load(f)
                print(f"  Train images: {len(train_data.get('images', []))}")
                print(f"  Train annotations: {len(train_data.get('annotations', []))}")
        
        if os.path.exists(val_ann_path):
            import json
            with open(val_ann_path, 'r') as f:
                val_data = json.load(f)
                print(f"  Val images: {len(val_data.get('images', []))}")
                print(f"  Val annotations: {len(val_data.get('annotations', []))}")
