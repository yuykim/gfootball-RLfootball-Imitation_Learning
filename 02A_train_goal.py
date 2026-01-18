import glob
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# 1. 모델 정의
# -------------------------
class BCModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# 2. 데이터 로더
# -------------------------
def load_npz_files(files):
    obs_list, act_list = [], []
    for fp in files:
        d = np.load(fp)
        obs_list.append(d["obs"])
        act_list.append(d["actions"])
    obs = np.concatenate(obs_list, axis=0)
    acts = np.concatenate(act_list, axis=0)
    return obs, acts

# -------------------------
# 3. main
# -------------------------
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "expert_dataset")
    goal_dir = os.path.join(root_dir, "goal")

    goal_files = sorted(glob.glob(os.path.join(goal_dir, "*.npz")))
    if len(goal_files) == 0:
        raise FileNotFoundError(f"goal 폴더에 npz가 없습니다: {goal_dir}")

    print(f"[GOAL ONLY] 파일 {len(goal_files)}개 로드 중...")
    obs_np, acts_np = load_npz_files(goal_files)

    states = torch.tensor(obs_np, dtype=torch.float32)
    actions = torch.tensor(acts_np, dtype=torch.long)

    num_classes = int(actions.max().item()) + 1
    print(f"프레임 수: {len(states)}, 액션 종류: {num_classes}")

    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # -------------------------
    # 4. 학습 설정
    # -------------------------
    model = BCModel(115, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("학습 시작...")
    for epoch in range(500):
        total_loss = 0.0
        for s, a in loader:
            optimizer.zero_grad()
            logits = model(s)
            loss = criterion(logits, a)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # -------------------------
    # 5. 저장
    # -------------------------
    save_path = os.path.join(current_dir, "il_model_goal_only.pth")
    torch.save(model.state_dict(), save_path)
    print(f"모델 저장 완료: {save_path}")

if __name__ == "__main__":
    main()
