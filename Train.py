import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import csv
import os

# 다른 파일에서 클래스 임포트
from Model import MultiTaskFATClassifier
from Dataset import MultiTaskFatDataset


def run_training_and_evaluation():
    # 데이터 로딩, 사용할 데이터셋 정의
    TRACE_DIR = '/feature_traces'  # <-- 학습시킬 .pt 파일이 있는 폴더 경로

    NUM_EPOCHS = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device : {device}")
    print(f"Data directory: {TRACE_DIR}")

    # 데이터셋 인스턴스화 및 분리
    full_dataset = MultiTaskFatDataset(TRACE_DIR)

    if len(full_dataset) == 0:
        print("데이터셋이 비어있어 학습을 진행할 수 없습니다.")
        return  # 함수 종료

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    if test_size == 0 and train_size > 0:
        test_size = 1
        train_size = len(full_dataset) - 1

    # 데이터셋이 1개뿐인 극단적인 경우 처리
    if train_size == 0 and test_size == 1:
        print("경고: 데이터가 1개뿐입니다. 학습이 불가합니다.")
        return

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print("\n가중 샘플러(WeightedRandomSampler) 생성 중...")

    # train_dataset (Subset 객체)에 포함된 샘플들의 '오류' 레이블(0 또는 1)만 추출
    # full_dataset.error_labels[train_dataset.indices] -> 훈련 세트의 실제 인덱스에 해당하는 레이블 목록
    train_labels = full_dataset.error_labels[train_dataset.indices]

    # 훈련 세트 내의 0(정상)과 1(오류) 개수 계산
    # .long()은 bincount를 위해 float를 정수로 변환
    class_counts = torch.bincount(train_labels.long())

    sampler = None
    shuffle = True

    if len(class_counts) < 2:
        print(f"경고: 훈련 세트에 클래스가 1개({len(class_counts)}개)만 존재합니다. 샘플러를 사용하지 않습니다.")
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    else:
        print(f"훈련 세트 클래스 수: 0(정상)={class_counts[0]}, 1(오류)={class_counts[1]}")

        # 각 클래스에 대한 가중치 계산 (1 / 클래스 샘플 수)
        # 예: 1/510000, 1/30000
        class_weights = 1. / class_counts.float()

        # 훈련 세트의 모든 샘플(46553개)에 대해 각자의 가중치를 할당
        # [0.000002, 0.000033, 0.000002, ... ]
        sample_weights = class_weights[train_labels.long()]

        # 가중 샘플러 정의
        # replacement=True: 중복 샘플링 허용 (오버샘플링)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        shuffle = False

        print("샘플러 생성 완료. DataLoader에 적용합니다.")

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, shuffle=shuffle, drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    #test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    #print(f"훈련 세트: {len(train_dataset)}개, 테스트 세트: {len(test_dataset)}개")
    print(f"훈련 세트: {len(train_dataset)}개 (가중 샘플링 적용), 테스트 세트: {len(test_dataset)}개")

    # --- 4. 모델 학습 ---
    print("\nStart training. . .")

    # 1. 모델, 옵티마이저 정의
    model = MultiTaskFATClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 1e-5)

    # 2. 두 개의 Loss 함수 정의
    criterion_class = nn.CrossEntropyLoss()
    criterion_error = nn.BCELoss()

    print("\n--- 다중 작업 모델(Multi-Task) 학습 시작 ---")

    for epoch in range(NUM_EPOCHS):
        #print("call model.train()..")
        model.train()
        #print("Done")

        #print("Set total_loss_c/e_raw to 0...")
        total_loss_c_raw = 0.0
        total_loss_e_raw = 0.0
        #print("Done")

        for traces, class_labels, _, error_labels, _, _ in train_loader:
            #print("Send traces to device")
            traces = traces.to(device)

            #print("Send class labels to device")
            class_labels = class_labels.to(device)

            #print("Send error labels to device")
            error_labels = error_labels.to(device)

            optimizer.zero_grad()
            #print("Done")

            # 모델 포워딩
            #print("Forwarding model. . .")
            class_outputs, error_outputs = model(traces)
            #print("Done")

            # Loss 계산
            #print("Calculating Loss. . .")
            loss_c = criterion_class(class_outputs, class_labels)

            #print("Clamping loss. . .")
            error_outputs_clamped = torch.clamp(error_outputs.squeeze(1), min=1e-7, max=1 - 1e-7)
            loss_e = criterion_error(error_outputs_clamped, error_labels)

            #print("Updating total loss. . .")
            total_loss = loss_c + loss_e

            #print("Backwarding loss. . .")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # [수정] .item()을 사용하여 순수 손실값만 누적
            #print("Accumulate total loss c/e. . .")
            total_loss_c_raw += loss_c.item()
            total_loss_e_raw += loss_e.item()
            #print("Done")

        # [수정] 에포크가 끝난 후, 배치 루프 밖에서 로그를 1회 출력합니다.
        avg_loss_c = total_loss_c_raw / len(train_loader)
        avg_loss_e = total_loss_e_raw / len(train_loader)

        print(f"[Epoch {epoch + 1}/{NUM_EPOCHS}] "
              f"Class Loss: {avg_loss_c:.4f} | "
              f"Error Loss: {avg_loss_e:.4f} | ")

    print("학습 완료.")

    MODEL_SAVE_PATH = "EDMN.pth"
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"훈련된 모델 가중치를 '{MODEL_SAVE_PATH}'에 저장했습니다.\n")
