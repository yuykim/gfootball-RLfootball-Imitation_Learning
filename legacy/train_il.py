import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# 1. 모델 정의
class BCModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

# 2. 데이터 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "expert_data.npz")
data = np.load(data_path)

states = torch.FloatTensor(data['obs'])
actions = torch.LongTensor(data['actions'])

num_classes = int(actions.max().item()) + 1 # 32개 액션 자동 대응
print(f"데이터 로드 완료. 액션 종류: {num_classes}개")

dataset = TensorDataset(states, actions)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3. 학습 설정
model = BCModel(115, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("학습 시작...")
for epoch in range(500): # 데이터가 적을 땐 epoch를 높이는 것이 유리
    total_loss = 0
    for s, a in loader:
        optimizer.zero_grad()
        loss = criterion(model(s), a)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# 4. 저장
save_path = os.path.join(current_dir, "il_model.pth")
torch.save(model.state_dict(), save_path)
print(f"모델 저장 완료: {save_path}")