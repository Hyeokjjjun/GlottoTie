import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """CNN Encoder with Dropout"""
    
    def __init__(self, input_channels=60, embedding_dim=512, dropout=0.3):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.7),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.7),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        
        return x


class PrototypicalNetwork(nn.Module):
    """Prototypical Network for 3-way K-shot Learning"""
    
    def __init__(self, encoder):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
    
    def forward(self, support, support_labels, query, n_way, k_shot):
        # Encode
        support_embeddings = self.encoder(support)
        query_embeddings = self.encoder(query)
        
        # Prototypes
        prototypes = self._compute_prototypes(
            support_embeddings, support_labels, n_way
        )
        
        # Logits
        logits = self._compute_logits(query_embeddings, prototypes)
        
        return logits
    
    def _compute_prototypes(self, embeddings, labels, n_way):
        prototypes = []
        for i in range(n_way):
            class_mask = (labels == i)
            class_embeddings = embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototype = F.normalize(prototype, p=2, dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def _compute_logits(self, query_embeddings, prototypes):
        distances = self._euclidean_distance(query_embeddings, prototypes)
        logits = -distances * 10  # Temperature scaling
        return logits
    
    def _euclidean_distance(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        
        distances = torch.pow(x - y, 2).sum(2)
        return distances