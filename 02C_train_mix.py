import glob
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

def load_one(fp):
    d = np.load(fp)
    return d["obs"], d["actions"]

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "expert_dataset")
    goal_dir = os.path.join(root_dir, "goal")
    nogoal_dir = os.path.join(root_dir, "no_goal")

    goal_files = sorted(glob.glob(os.path.join(goal_dir, "*.npz")))
    nogoal_files = sorted(glob.glob(os.path.join(nogoal_dir, "*.npz")))

    if len(goal_files) == 0 or len(nogoal_files) == 0:
        raise FileNotFoundError("goal/no_goal 폴더 둘 다 npz가 있어야 70/30 혼합이 가능합니다.")

    # ✅ 혼합 비율 (에피소드 파일 단위)
    p_goal = 0.7

    # ✅ 몇 개의 에피소드를 섞을지 (둘 중 작은쪽에 맞추는 게 안전)
    # 예: goal이 50개, nogoal이 20개면 → 20개 수준에서 맞춰야 함
    base = min(len(goal_files), len(nogoal_files))
    total_episodes = base * 2  # 원하는 만큼 조절 가능

    n_goal = int(total_episodes * p_goal)
    n_nogoal = total_episodes - n_goal

    sampled_goal = random.sample(goal_files, k=min(n_goal, len(goal_files)))
    sampled_nogoal = random.sample(nogoal_files, k=min(n_nogoal, len(nogoal_files)))

    print(f"[MIX 70/30] goal ep={len(sampled_goal)}, no_goal ep={len(sampled_nogoal)} 로드")

    obs_list, act_list = [], []
    for fp in sampled_goal + sampled_nogoal:
        o, a = load_one(fp)
        obs_list.append(o)
        act_list.append(a)

    obs_np = np.concatenate(obs_list, axis=0)
    acts_np = np.concatenate(act_list, axis=0)

    states = torch.tensor(obs_np, dtype=torch.float32)
    actions = torch.tensor(acts_np, dtype=torch.long)

    num_classes = int(actions.max().item()) + 1
    print(f"프레임 수: {len(states)}, 액션 종류: {num_classes}")

    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = BCModel(115, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("학습 시작...")
    for epoch in range(500):
        total_loss = 0.0
        for s, a in loader:
            optimizer.zero_grad()
            loss = criterion(model(s), a)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    save_path = os.path.join(current_dir, "il_model_mix.pth")
    torch.save(model.state_dict(), save_path)
    print(f"모델 저장 완료: {save_path}")

if __name__ == "__main__":
    main()