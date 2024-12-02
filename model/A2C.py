import torch
import torch.nn as nn
import torch.nn.functional as F

class A2C(nn.Module):
    def __init__(self, name: str, obs_space, action_space):
        super(A2C, self).__init__()
        self.observation = obs_space
        self.action = action_space
        self.scope_name = name

        # 定义输入层
        self.inputs = nn.Linear(self.observation.shape[0], 16)

        
        self.build_network()

    def build_network(self):
        # Critic Network
        self.critic_fc1 = nn.Linear(16, 32)
        self.critic_fc2 = nn.Linear(32, 1)

        # Actor Network
        self.actor_fc1 = nn.Linear(16, 16)
        self.actor_fc2 = nn.Linear(16, 32)
        self.actor_fc3 = nn.Linear(32, 16)
        self.actor_out = nn.Linear(16, self.action.n)

    def forward(self, x):
        x = F.tanh(self.inputs(x))

        # 评论家网络前向传播
        value = F.tanh(self.critic_fc1(x))
        value = self.critic_fc2(value)

        # 演员网络前向传播
        aout = F.tanh(self.actor_fc1(x))
        aout = F.tanh(self.actor_fc2(aout))
        aout = F.tanh(self.actor_fc3(aout))
        act_probs = F.softmax(self.actor_out(aout), dim=-1)

        return act_probs, value

    # 根据给定输入获取动作
    def get_action(self, inputs):
        with torch.no_grad():
            act_probs, _ = self.forward(inputs)
            act = torch.multinomial(act_probs, 1)
            return act.squeeze().cpu().numpy()

    # 获取给定输入的价值预测
    def get_value(self, inputs):
        with torch.no_grad():
            _, value = self.forward(inputs)
            return value.squeeze().cpu().numpy()

    # 获取策略更新所需的所有可训练变量
    def trainable_vars(self):
        return [param for param in self.parameters()]
