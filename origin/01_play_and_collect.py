import gym
import gfootball.env as football_env
from gfootball.env.players.gamepad import Player as GamepadPlayer
from gfootball.env.players.keyboard import Player as KeyboardPlayer
from gfootball.env import football_action_set
import numpy as np
import os
import pygame
from datetime import datetime

# ---------------------------
# 1. 저장 폴더 세팅
# ---------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.join(current_dir, "expert_dataset")
goal_dir = os.path.join(root_dir, "goal")       # 골 넣은 에피소드
nogoal_dir = os.path.join(root_dir, "no_goal")  # 골 못 넣은 에피소드

os.makedirs(goal_dir, exist_ok=True)
os.makedirs(nogoal_dir, exist_ok=True)

run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")  # 실행마다 구분용

# ---------------------------
# 2. 환경 설정
# academy_3_vs_1_with_keeper - (우리팀) 공격수 3명, (상대) 수비수 1명, 골키퍼 1명
# academy_corner - 코너킥 상황 
# academy_counterattack_easy - 역습 상황
# academy_pass_and_shoot_with_keeper - (우리팀) 공격수 2명, (상대) 수비수 1명, 골키퍼 1명
# academy_run_to_score_with_keeper - 뒤에서 수비수가 달려오고 골키퍼랑 1대1 상황
# ---------------------------
env_config = {'action_set': 'full'}
env = football_env.create_environment(
    env_name='academy_pass_and_shoot_with_keeper',
    representation='simple115v2',
    render=True
)

# ---------------------------
# 3. 조작 장치 선택
# ---------------------------
player_config = {'player_gamepad': 0, 'left_players': 1, 'right_players': 0}
controller = GamepadPlayer(player_config, env_config)

# 키보드 사용 시:
# player_config = {'left_players': 1, 'right_players': 0}
# controller = KeyboardPlayer(player_config, env_config)

all_actions = football_action_set.get_action_set(env_config)

clock = pygame.time.Clock()
TARGET_FPS = 18

print(f"start data collect (FPS: {TARGET_FPS})")
print(f"save dir : {root_dir}")

# ---------------------------
# 4. 에피소드 단위 버퍼
# ---------------------------
episode_id = 0
ep_obs, ep_actions = [], []
ep_reward_sum = 0.0

def save_episode(ep_obs, ep_actions, ep_reward_sum, episode_id):
    # ✅ 성공 판정(가장 단순/안정): 에피소드 총 보상 > 0 이면 골 성공
    # (대부분 scoring reward가 골=+1로 구성됨) :contentReference[oaicite:1]{index=1}
    success = (ep_reward_sum > 0)

    target_dir = goal_dir if success else nogoal_dir
    fname = f"{run_tag}_ep{episode_id:06d}_R{ep_reward_sum:.2f}.npz"
    fpath = os.path.join(target_dir, fname)

    np.savez_compressed(
        fpath,
        obs=np.asarray(ep_obs),
        actions=np.asarray(ep_actions, dtype=np.int32),
        episode_reward=np.float32(ep_reward_sum),
        success=np.bool_(success),
        episode_id=np.int32(episode_id),
    )
    return success, fpath

try:
    obs = env.reset()

    while True:
        clock.tick(TARGET_FPS)

        # 액션 선택
        action = controller.take_action([obs])
        action_int = all_actions.index(action)

        # 버퍼 저장 (step 전 obs 기준)
        ep_obs.append(obs)
        ep_actions.append(action_int)

        # env step
        obs, reward, done, info = env.step(action)
        ep_reward_sum += float(reward)

        if done:
            episode_id += 1
            success, fpath = save_episode(ep_obs, ep_actions, ep_reward_sum, episode_id)

            print(
                f"[EP {episode_id}] {'GOAL ✅' if success else 'NO GOAL ❌'} | "
                f"steps={len(ep_obs)} | R={ep_reward_sum:.2f} | saved -> {os.path.basename(fpath)}"
            )

            # 다음 에피소드 초기화
            ep_obs, ep_actions = [], []
            ep_reward_sum = 0.0
            obs = env.reset()

except KeyboardInterrupt:
    print("\ninterrupt!!")

finally:
    env.close()
