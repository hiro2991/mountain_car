import numpy as np
import gym
from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from gym import wrappers  # gymの画像保存
from keras import backend as K
import tensorflow as tf
import time

######## mountain car の条件設定 ########
#状態は、position, velocity
#行動は、0:left、1:no push、2:right
#報酬は、各自毎に-1を付与、ゴール(0.5)につくまで
#開始位置は、-0.6～-0.4でランダム
#ステップ200かゴールで1エピソード終了


def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    print("test")
    return K.mean(loss)

# Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate = 0.01, state_size = 2, action_size = 3, hidden_size = 10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size)) #kerasのdenseは全結合層
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self.model.compile(loss = huberloss, optimizer = self.optimizer)
        self.count = 0

    # 重みの学習
    def replay(self, memory, batch_size, gamma, main_QN):
        inputs = np.zeros((batch_size, 2))
        targets = np.zeros((batch_size, 3))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            st_time = time.time()

            inputs[i:i + 1] = state_b
            target = reward_b

#            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算
            retmainQs = self.model.predict(next_state_b)[0]

            next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
            target = reward_b + gamma * main_QN.model.predict(next_state_b)[0][next_action]
            #print("reward_b:{}, mod_reward:{}".format(reward_b, gamma * main_QN.model.predict(next_state_b)[0][next_action]))

            targets[i] = self.model.predict(state_b)    # Qネットワークの出力
            targets[i][action_b] = target               # 教師信号

        self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定


# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)


#探索判定(エピソード数が浅い時はランダム、多い時は最適)
def judge_action(next_state, episode, mainQN):
    #e-greedy
    epsilon = 0.5*(1 / (episode+1))
    if epsilon <= np.random.uniform(0, 1):
            retTargetQs = mainQN.model.predict(next_state)[0]
            next_action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
    else:
        next_action = np.random.choice([0, 1, 2])

    return next_action



if __name__ == "__main__":
    env = gym.make("MountainCar-v0") #環境作成
    #env = wrappers.Monitor(env, "./", video_callable = (lambda episode: episode % 1000 == 0 ))

    max_episode = 10000
    max_steps = 200

    #学習率
    alpha = 0.001 #0.1くらいが推奨？
    #割引率
    gamma = 0.99 #0.9～0.99が推奨？1だと収束しない。0だと将来の報酬は考慮しない。

    hidden_size = 16               # Q-networkの隠れ層のニューロンの数
    memory_size = 10000            # バッファーメモリの大きさ
    batch_size = 32

    #モデル作成
    main_QN = QNetwork(hidden_size = hidden_size, learning_rate = alpha)
    memory = Memory(max_size = memory_size)

    #エピソードループ
    for episode in range(max_episode):

        #初期化
        observation = env.reset() #状態初期化
        action = np.random.choice([0,1,2]) #初期アクションを乱数で決定
        episode_reward = 0 #エピソードの報酬を初期化

        observation = np.reshape(observation, [1,2])

        #ステップループ
        for step in range(max_steps):
            #1.状態更新
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1,2])

            #2.報酬設定・更新
            episode_reward = episode_reward + reward
            #3.学習用データ追加
            memory.add((observation, action, reward, next_state))

            #4.Qtable更新
            if memory.len() > batch_size:
                main_QN.replay(memory, batch_size, gamma, main_QN)

            #5.行動の決定
            action = judge_action(next_state, episode, main_QN)

            #6.状態の更新
            observation = next_state
            #7.終了判定
            if done or step == max_steps - 1:
                print("episode {} is finished".format(episode))
                print("total reward is {}".format(episode_reward))
                if episode_reward != -200:
#                    print("episode {} is finished".format(episode))
                    print("###goal!###")
#                elif episode % 100 == 0:
#                    print("episode {} is finished".format(episode))

                break



