import torch
import numpy as np
import os
import random
from glob import glob

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class StrongAugmentation:
    """Data augmentation"""
    
    def __init__(self, prob=0.8):
        self.prob = prob
    
    def __call__(self, data):
        """
        Args:
            data: (60, 516, 85) tensor
        """
        # 1. Gaussian Noise
        if random.random() < self.prob:
            noise = torch.randn_like(data) * 0.02
            data = data + noise
            data = torch.clamp(data, 0, 1)
        
        # 2. Channel-wise scaling
        if random.random() < self.prob:
            scale = torch.randn(60, 1, 1) * 0.1 + 1.0
            scale = torch.clamp(scale, 0.8, 1.2)
            data = data * scale
            data = torch.clamp(data, 0, 1)
        
        # 3. Horizontal shift
        if random.random() < self.prob:
            shift = random.randint(-10, 10)
            data = torch.roll(data, shifts=shift, dims=2)
        
        return data


class FewShotDataset:
    """Generate 3-way 2-shot episodes from commercial data"""
    
    def __init__(self, data_path, augmentation=None, num_classes=3):
        self.data_path = data_path
        self.augmentation = augmentation
        self.num_classes = num_classes
        
        # Load .npy file paths for each class
        self.class_to_files = {}
        class_names = sorted(os.listdir(data_path))

        # Use only the first 3 classes
        class_names = class_names[:num_classes]
        
        for class_name in class_names:
            class_path = os.path.join(data_path, class_name)
            if os.path.isdir(class_path):
                npy_files = glob(os.path.join(class_path, '*.npy'))
                if len(npy_files) > 0:
                    self.class_to_files[class_name] = npy_files
        
        self.classes = sorted(list(self.class_to_files.keys()))
        
        print(f"=== Commercial Dataset (3-class) ===")
        print(f"Total classes: {len(self.classes)}")
        for cls in self.classes:
            print(f"  {cls}: {len(self.class_to_files[cls])} samples")
    
    def sample_episode(self, n_way=3, k_shot=2, n_query=15):
        """
        Sample a 3-way 2-shot episode

        Returns:
            support_data: (n_way * k_shot, 60, 516, 85)
            support_labels: (n_way * k_shot,)
            query_data: (n_way * n_query, 60, 516, 85)
            query_labels: (n_way * n_query,)
        """
        selected_classes = self.classes  # Use all 3 classes
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for label_idx, class_name in enumerate(selected_classes):
            class_files = self.class_to_files[class_name]
            
            # Sample (k_shot + n_query) items
            n_samples = k_shot + n_query
            if len(class_files) < n_samples:
                sampled = random.choices(class_files, k=n_samples)
            else:
                sampled = random.sample(class_files, n_samples)
            
            # Support set
            for file_path in sampled[:k_shot]:
                data = self.load_data(file_path)
                support_data.append(data)
                support_labels.append(label_idx)
            
            # Query set
            for file_path in sampled[k_shot:]:
                data = self.load_data(file_path)
                query_data.append(data)
                query_labels.append(label_idx)
        
        # Convert to tensors
        support_data = torch.stack(support_data)
        support_labels = torch.LongTensor(support_labels)
        query_data = torch.stack(query_data)
        query_labels = torch.LongTensor(query_labels)

        return support_data, support_labels, query_data, query_labels

    def load_data(self, file_path):
        """
        Load a .npy file and convert it to a tensor
        """
        data = np.load(file_path)  # (516, 85, 60)
        data = np.transpose(data, (2, 0, 1))  # (60, 516, 85)
        tensor = torch.FloatTensor(data)

        # Apply augmentation
        if self.augmentation is not None:
            tensor = self.augmentation(tensor)
        
        return tensor


class OurDeviceDataset:
    """Our device data"""

    def __init__(self, data_path, num_classes=3):
        self.data_path = data_path
        self.num_classes = num_classes

        # Load .npy files for each class
        self.class_to_files = {}
        class_names = sorted(os.listdir(data_path))
        class_names = class_names[:num_classes]
        
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(data_path, class_name)
            if os.path.isdir(class_path):
                npy_files = sorted(glob(os.path.join(class_path, '*.npy')))
                self.class_to_files[class_idx] = npy_files
        
        print(f"\n=== Our Device Dataset (3-class) ===")
        print(f"Total classes: {len(self.class_to_files)}")
        for cls_idx, files in self.class_to_files.items():
            print(f"  Class {cls_idx}: {len(files)} samples")
    
    def get_test_episode(self, k_shot=2):
        """Generate a 3-way k-shot test episode"""
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for class_idx in range(self.num_classes):
            files = self.class_to_files[class_idx]
            
            # Support: first k_shot items
            for file_path in files[:k_shot]:
                data = self.load_data(file_path)
                support_data.append(data)
                support_labels.append(class_idx)

            # Query: the rest
            for file_path in files[k_shot:]:
                data = self.load_data(file_path)
                query_data.append(data)
                query_labels.append(class_idx)
        
        support_data = torch.stack(support_data)
        support_labels = torch.LongTensor(support_labels)
        query_data = torch.stack(query_data)
        query_labels = torch.LongTensor(query_labels)
        
        return support_data, support_labels, query_data, query_labels
    
    def get_cross_validation_splits(self, k_shot=2, n_folds=3):
        """Select a different support set for each fold"""
        splits = []
        
        for fold in range(n_folds):
            support_data = []
            support_labels = []
            query_data = []
            query_labels = []
            
            support_start = fold
            
            for class_idx in range(self.num_classes):
                files = self.class_to_files[class_idx]
                n_samples = len(files)
                
                # Support indices
                support_indices = []
                for i in range(k_shot):
                    idx = (support_start + i) % n_samples
                    support_indices.append(idx)

                # Support data
                for idx in support_indices:
                    data = self.load_data(files[idx])
                    support_data.append(data)
                    support_labels.append(class_idx)

                # Query: the rest
                for idx, file_path in enumerate(files):
                    if idx not in support_indices:
                        data = self.load_data(file_path)
                        query_data.append(data)
                        query_labels.append(class_idx)
            
            support = torch.stack(support_data)
            support_lbl = torch.LongTensor(support_labels)
            query = torch.stack(query_data)
            query_lbl = torch.LongTensor(query_labels)
            
            splits.append((support, support_lbl, query, query_lbl))
        
        return splits
    
    def load_data(self, file_path):
        """Load a file"""
        data = np.load(file_path)
        data = np.transpose(data, (2, 0, 1))
        tensor = torch.FloatTensor(data)
        return tensor