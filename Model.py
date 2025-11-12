import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskFATClassifier(nn.Module):
    """
    FAT 트레이스를 입력받아 두 가지를 동시에 예측 (자동 손실 균형 적용)
    1. 원본 클래스 (0-9)
    2. 오류 발생 확률 (0-1)
    """

    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=10):
        super(MultiTaskFATClassifier, self).__init__()

        BN_EPS = 1e-6

        # 은닉층 (1024 - 256 - 256 - 10+1)
        #self.bn_input = nn.BatchNorm1d(input_dim, eps=BN_EPS)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.bn1 = nn.BatchNorm1d(hidden_dim, eps=BN_EPS)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.bn2 = nn.BatchNorm1d(hidden_dim, eps=BN_EPS)

        # 출력 헤드 (클래스 + 오류 확률)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.error_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x의 shape: (Batch, 1024)
        #x = self.bn_input(x)
        x = F.relu(self.fc1(x))
        x_features = F.relu(self.fc2(x)) # (Batch, 256)

        class_output = self.class_head(x_features)
        error_prob = torch.sigmoid(self.error_head(x_features))

        # 출력 + 손실 가중치 두 개 반환(클래스 / 오류 확률)
        return class_output, error_prob #, self.log_var_c, self.log_var_e
