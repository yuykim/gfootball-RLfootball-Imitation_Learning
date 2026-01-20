import torch
import torch.nn as nn
import gfootball.env as football_env
from gfootball.env import football_action_set
import numpy as np
import time
import os
import sys

sys.path.append("..")

import utils

class BCModel(nn.Module):
    def __init__(self, output_dim):
        super(BCModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(115, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

def test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "il_model_goal_only.pth")
    
    # 1. 모델 로드 및 크기 자동 감지
    checkpoint = torch.load(model_path, map_location='cpu')
    output_dim = checkpoint['net.4.bias'].shape[0]
    model = BCModel(output_dim)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. 액션 리스트 및 환경 생성
    all_actions = football_action_set.get_action_set({'action_set': 'full'})
    env = football_env.create_environment(env_name='academy_corner', 
                                          representation='simple115v2', 
                                          render=False
                                          )

    print("start!!")
    obs = env.reset()
    goal_count = 0
    game_count = 0
    game = True
    t = 0
    try:
        while game:
            with torch.no_grad():
                output = model(torch.FloatTensor(obs).unsqueeze(0))
                # 확률 기반 샘플링 (선수들이 더 역동적으로 움직임)
                probs = torch.softmax(output, dim=1)
                action_idx = torch.distributions.Categorical(probs).sample().item()
            
            # 인덱스를 액션 객체로 변환하여 에러 방지
            obs, reward, done, info = env.step(all_actions[action_idx])

            frame = env.render(mode="rgb_array")
            utils.save_frame(frame, t)
            t = t+1

            time.sleep(0.01)
            if(reward > 0.1):
                goal_count += 1
            if done: 
                obs = env.reset()
                print(f"score : {goal_count}/{game_count+1}")
                game_count += 1
            if game_count == 1: 
                game = False
    finally:
        print("done.")
        env.close()

if __name__ == "__main__":
    utils.cleanup()
    test()
    utils.make_video()