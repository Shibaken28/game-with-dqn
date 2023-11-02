"""
to do
- rewaradの平均も出力するようにする
"""



import copy
from collections import deque
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# グラフ描画
import matplotlib.pyplot as plt
# 日付取得
import datetime
# csv出力
import csv
# ディレクトリ作成
import os


# この実行時のidを生成(ランダムな文字列)
import string
id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
print(id)
# ディレクトリ作成
os.mkdir("./"+id)
os.mkdir("./"+id+"/img")
os.mkdir("./"+id+"/csv")


class Game:
    """
    200カウント以内に敵に当たらなければ成功
    state:
        敵の位置と前フレームの場所 (x,y) (x',y')
        自機の位置 (x,y)
    action:
        右に行く(x+1), 左に行く(x-1), そのまま(x)
    reward:
        敵に当たったら-1, それ以外は0
        ゲーム終了時に成功なら+1
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        # 敵の軌道
        self.game_length = 50
        self.width = 100
        self.height = 100
        self.enemy_pos_list = []
        self.time = 1
        # enemy_pos[i] = 時刻iにいる場所
        # 移動方法は数通りから選ぶ
        # 0: まっすぐ、1: 斜め右に、2: 斜め左に
        p = random.randint(0, 2)
        if p == 0:
            x = random.randint(0, self.width)
            for t in range(self.game_length+1):
                self.enemy_pos_list.append((x, t*2))
        elif p == 1:
            x = random.randint(0, self.width) / 2
            for t in range(self.game_length+1):
                self.enemy_pos_list.append((x+t, t*2))
        else:
            x = random.randint(0, self.width) / 2 + self.width / 2
            for t in range(self.game_length+1):
                self.enemy_pos_list.append((x-t, t*2))


        # 自機の初期位置
        self.player_pos = (self.width/2, self.height//4*3)
        self.player_size = self.width//10
        self.enemy_pos = self.enemy_pos_list[0]
        self.enemy_size = self.width//10
        return self.states(), 0

    def states(self):
        # np.array([self.enemy_pos_list[self.time][0], self.enemy_pos_list[self.time][1], self.enemy_pos_list[self.time-1][0], self.enemy_pos_list[self.time-1][1], self.player_pos[0], self.player_pos[1]])
        # float32のTensorに変換
        r = self.time-5
        if r < 0:
            r = 0
        return torch.tensor([self.enemy_pos_list[self.time][0]/self.width, self.enemy_pos_list[self.time][1]/self.height, self.enemy_pos_list[r][0]/self.width, self.enemy_pos_list[r][1]/self.height, self.player_pos[0]/self.width, self.player_pos[1]/self.height], dtype=torch.float32)


    def step(self, action):
        self.time += 1
        # actionを実行
        reward = 0
        if action == 0:
            self.player_pos = (self.player_pos[0]-1, self.player_pos[1])
        elif action == 1:
            self.player_pos = (self.player_pos[0]+1, self.player_pos[1])
        else:
            # 動かないとえらい
            reward = 0.1
            pass
        # 敵の位置を更新

        self.enemy_pos = self.enemy_pos_list[self.time]
        # 報酬を計算
        done = False
        win = False
        if self.isCollision():
            reward = -1
            done = True
        # 画面外
        elif self.player_pos[0] < 0 or self.player_pos[0] > self.width:
            reward = -1
            done = True
        else:
            # ゲームクリア
            if self.time == self.game_length-1:
                reward = 1
                done = True
                win = True

        return self.states(), reward, done, {}, win



    def isCollision(self):
        # 敵と自機の距離を計算
        distance = np.sqrt((self.enemy_pos[0]-self.player_pos[0])**2 + (self.enemy_pos[1]-self.player_pos[1])**2)
        if distance < self.player_size + self.enemy_size:
            return True
        else:
            return False



class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int64))
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(6, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.95
        self.lr = 0.0005
        self.epsilon = 0.05
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 3

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state_tensor = torch.tensor(state).unsqueeze(0)  # バッチ次元を追加
            qs = self.qnet(state_tensor)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(len(action)), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]

        next_q.detach()
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


# 途中経過を保存
def draw_graph(result):
    # i回目までの勝率   
    rate = []
    sum = 0
    for i in range(len(result)):
        if result[i]:
            sum += 1
        rate.append(sum/(i+1))

    plt.plot(rate)
    plt.xlabel('episode')
    plt.ylabel('rate')
    # y軸の範囲を0~1にする
    plt.ylim(0, 1)
    # 現在の日付+時間+エピソード数をファイル名にする
    now = datetime.datetime.now()
    path = './'+id+'/img/'
    # YYYYMMDD_HHMMSS_episode.png
    filename = "result_"+id+"_"+now.strftime('%Y%m%d_%H%M%S') + '_' + str(len(result)) + '.png'
    print(filename)
    plt.savefig(path + filename)
    plt.close()

def output_csv(state_history, episode):
    # 現在の日付+時間+エピソード数をファイル名にする
    now = datetime.datetime.now()
    path = './'+id+'/csv/'
    # YYYYMMDD_HHMMSS_episode.csv
    filename = "result_"+id+"_"+now.strftime('%Y%m%d_%H%M%S') + '_' + str(episode) + '.csv'
    print(filename)
    # csvファイルを開く
    with open(path + filename, 'w') as f:
        # ヘッダーを指定する
        writer = csv.writer(f, lineterminator='\n')
        # i番目の要素をi行目に出力
        for i in range(len(state_history)):
            # tensorをただのlistに変換
            state = state_history[i].tolist()
            writer.writerow(state)

        
    # ファイルを閉じる
    f.close()
    print("save csv")    



episodes = 100000
sync_interval = 50
env = Game()
agent = DQNAgent()
reward_history = []
result = []

# ビジュアライズするために状態を保存

for episode in range(episodes):
    state_history = []
    state, _ = env.reset()
    done = False
    total_reward = 0
    win = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info, win  = env.step(action)
        state_history.append(state)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()

    result.append(win)
    reward_history.append(total_reward)
    print("episode :{}, win : {}, total reward : {}".format(episode, win, total_reward))
    # 平方数の時にグラフを描画
    r = int(np.sqrt(episode))
    if r*r == episode:
        draw_graph(result)
        output_csv(state_history, episode)
        