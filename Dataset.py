import os
import glob
import torch
from torch.utils.data import Dataset

class MultiTaskFatDataset(Dataset):
    def __init__(self, trace_dir):
        self.traces = []
        self.class_labels = []  # (Target 1: 0~9)
        self.predicted_labels = []  # 예측된 클래스
        self.error_labels = []  # (Target 2: 0 or 1)
        self.filenames = []  # (CSV용: 파일명)
        self.type_strings = []  # (CSV용: 'Type1', 'Type2', 'Type3')


        print("데이터 로딩 및 전처리 시작...")

        file_paths = glob.glob(os.path.join(trace_dir, "*.pt"))
        if not file_paths:
            print(f"경고: '{trace_dir}' 경로에 *.pt 파일이 없습니다.")
            return

        temp_data_list = []

        for f_path in file_paths:
            filename = os.path.basename(f_path)
            try:

                # 1. 데이터 로드
                # CPU로 로드하여 장치 호환성 문제 방지
                loaded_data = torch.load(f_path, map_location=torch.device('cpu'))

                original_label = loaded_data['label']
                trace_tensor = loaded_data['trace']  # (1, 1024)
                prediction = loaded_data['prediction']
                error_type = loaded_data['type']

                if error_type == 0:
                    type_str = 'Type1'
                else :
                    type_str = 'Type2'

                temp_data_list.append(trace_tensor.squeeze(0))  # (1024,)
                self.class_labels.append(original_label)
                self.error_labels.append(error_type)
                self.filenames.append(filename)
                self.type_strings.append(type_str)
                self.predicted_labels.append(prediction)

            except Exception as e:
                print(f"'{filename}' 처리 중 오류: {e}")

        # 정규화 (전체 데이터 대상)
        if temp_data_list:
            all_traces_tensor = torch.stack(temp_data_list)  # (N, 1024)
            mean = all_traces_tensor.mean(dim=0, keepdim=True)
            std = all_traces_tensor.std(dim=0, keepdim=True)
            std[std == 0] = 1e-6  # 0으로 나누기 방지

            self.traces = (all_traces_tensor - mean) / std  # 정규화된 데이터 저장
            self.class_labels = torch.tensor(self.class_labels, dtype=torch.long)
            self.error_labels = torch.tensor(self.error_labels, dtype=torch.float)  # BCELoss를 위해 float
            print(f"총 {len(self.traces)}개의 데이터 로드 및 정규화 완료.")
        else:
            print("처리할 데이터를 찾을 수 없습니다.")

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        # 5개 항목 반환
        return (
            self.traces[idx],
            self.class_labels[idx],
            self.predicted_labels[idx],
            self.error_labels[idx],
            self.filenames[idx],
            self.type_strings[idx]
        )
