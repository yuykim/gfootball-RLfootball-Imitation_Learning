import gym
import gfootball.env as football_env
# 조작 장치 임포트
from gfootball.env.players.gamepad import Player as GamepadPlayer
from gfootball.env.players.keyboard import Player as KeyboardPlayer
from gfootball.env import football_action_set
import numpy as np
import os
import pygame

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "expert_data.npz")

# 2. 환경 설정
env_config = {'action_set': 'full'}
env = football_env.create_environment(
    env_name='academy_pass_and_shoot_with_keeper',
    representation='simple115v2', 
    render=True
)

# 3. 조작 장치 선택 
# -----------------------------------------------------------------
# 게임패드 사용
player_config = {'player_gamepad': 0, 'left_players': 1, 'right_players': 0}
controller = GamepadPlayer(player_config, env_config)

# 키보드 사용 
# player_config = {'left_players': 1, 'right_players': 0}
# controller = KeyboardPlayer(player_config, env_config)
# -----------------------------------------------------------------

all_actions = football_action_set.get_action_set(env_config)

clock = pygame.time.Clock()
TARGET_FPS = 18 # 10은 느리고 30은 빠를 때의 최적 속도

obs_buffer, action_buffer = [], []

print(f"데이터 수집 시작! (현재 설정 FPS: {TARGET_FPS})")

try:
    obs = env.reset()
    while True:
        clock.tick(TARGET_FPS)
        
        # 현재 조작기(패드 또는 키보드)로부터 액션 객체 수신
        action = controller.take_action([obs])
        
        # 액션 객체를 숫자로 변환하여 저장
        action_int = all_actions.index(action)
        
        obs_buffer.append(obs)
        action_buffer.append(action_int)
        
        # 환경에 액션 전달
        obs, reward, done, info = env.step(action)
        
        if done:
            obs = env.reset()
            print(f"\n[안내] 경기 종료! 현재까지 {len(obs_buffer)} 프레임 확보됨.")

        if len(obs_buffer) % 1000 == 0:
            print(f"기록 중... {len(obs_buffer)} 프레임 돌파")

except KeyboardInterrupt:
    if len(obs_buffer) > 0:
        # 데이터 저장
        np.savez(save_path, obs=np.array(obs_buffer), actions=np.array(action_buffer))
        print(f"\n저장 성공! 파일 위치: {save_path}")
        print(f"총 수집 프레임: {len(obs_buffer)}")
    else:
        print("\n저장할 데이터가 없습니다.")
finally:
    env.close()