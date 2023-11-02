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
    puyopuyo
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        # 50ターン生きれば勝ち
        # 盤面のサイズ
        self.width = 6
        self.height = 12
        # 0:空白, 1:赤, 2:青, 3:緑, 4:黄 の4色
        self.board = np.zeros((self.height, self.width))
        # 50ターンで降るぷよの色を決定する
        self.puyo_list = []
        for i in range(50):
            self.puyo_list.append((random.randint(1, 4), random.randint(1, 4)))

    def states(self):
        
        return torch.tensor([self.enemy_pos_list[self.time][0]/self.width, self.enemy_pos_list[self.time][1]/self.height, self.enemy_pos_list[r][0]/self.width, self.enemy_pos_list[r][1]/self.height, self.player_pos[0]/self.width, self.player_pos[1]/self.height], dtype=torch.float32)


    def step(self, action):
        

        return self.states(), reward, done, {}, win






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
        