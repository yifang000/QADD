import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from baselines.common import schedules
from Env_molecule import *
from Get_action import get_valid_actions as getAction

def parse_arguments_RL(parser):
    #parser.add_argument('--device', type=str,choices=['cpu','gpu'], default='cpu')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--target_iter', type=int, default=100)
    parser.add_argument('--memory_capacity', type=int, default=2000)
    parser.add_argument('--state_dim', type=int, default=2048)
    parser.add_argument('--action_dim', type=int, default=2048)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--max_episode', type=int, default=5000)
    parser.add_argument('--max_step', type=int, default=60)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--target_mol', type=str, default='c1ccccc1')
    parser.add_argument('--output_file', type=str, default='../log/result')
    args = parser.parse_args()
    return args
    
parser = argparse.ArgumentParser()
args = parse_arguments_RL(parser)

TARGET_REPLACE_ITER = args.target_iter
####baselines epsilon decline : t=0,epsilon=1.0 -> t=MAX_episode/2,epsilon=0.1 -> t=MAX_episode,epsilon=0.01
epsilon = schedules.PiecewiseSchedule([(0, 1.0), (int(args.max_episode/2), 0.1),(args.max_episode, 0.01)],outside_value=0.01)
#an example weighted multi-object(single Q network)
weight = schedules.PiecewiseSchedule([(0, 0.99), (int(args.max_episode/2), 0.5),(args.max_episode, 0.1)],outside_value=0.1)

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(args.state_dim , 1024)
        self.fc1.weight.data.normal_(0,0.01)
        #self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024 , 512)
        self.fc2.weight.data.normal_(0,0.01)
        #self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512 , 128)
        self.fc3.weight.data.normal_(0,0.01)
        #self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128 , 32)
        self.fc4.weight.data.normal_(0,0.01)
        #self.bn4 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, args.output_dim)
        
    def forward(self, x):
        #x = F.relu(self.bn1(self.fc1(x)))
        #x = F.relu(self.bn2(self.fc2(x)))
        #x = F.relu(self.bn3(self.fc3(x)))
        #x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        value = self.out(x)
        return value


class Memory(object):
    def __init__(self,capacity): 
        self.capacity = capacity
        self.dim = args.state_dim + args.action_dim + 1
        self.data = np.zeros((self.capacity, self.dim))
        self.pointer = 0 
    def store(self,s,a,r):
        index = self.pointer % self.capacity
        self.data[index] = np.concatenate((s, a, [r]),axis=0)
        self.pointer += 1
    def sample(self,n):
        assert self.pointer >= self.capacity, 'Memory not full' 
        indexs = np.random.choice(self.capacity,size = n)
        return self.data[indexs],indexs

class multi_Memory(object):
    def __init__(self,capacity,reward_dim): 
        self.capacity = capacity
        self.dim = args.state_dim + args.action_dim + reward_dim
        self.data = np.zeros((self.capacity, self.dim))
        self.pointer = 0
    def store(self,s,a,r):
        index = self.pointer % self.capacity
        self.data[index] = np.concatenate((s, a, r),axis=0)
        self.pointer += 1
    def sample(self,n):
        assert self.pointer >= self.capacity, 'Memory not full'
        indexs = np.random.choice(self.capacity,size = n)
        return self.data[indexs],indexs

class Memory_smiles(object):
    def __init__(self,capacity): 
        self.capacity = capacity
        self.data = [i for i in range(capacity)]
        self.pointer = 0 
    def store(self,a):
        index = self.pointer % self.capacity 
        self.data[index] = a
        self.pointer += 1
    def sample(self,indexs):
        return [self.data[i] for i in indexs]
    

class DQN(object):
    def __init__(self):
        self.eval_Q = Net() #eval Q network 
        self.target_Q = Net() #target Q network
        self.step_counter = 0 
        self.optimizer = torch.optim.Adam(self.eval_Q.parameters(),lr=args.lr)
        self.loss_function = nn.MSELoss() #MSE loss = (y_i - Q(s_t,a^{'},w^{-}'))^2
    def choose_action(self,state): 
        actions = getAction(state,atom_types = ['C','N','O'],
                            allow_removal = True,
                            allow_no_modification = True,
                            allowed_ring_sizes = (3,4,5,6),
                            allow_bonds_between_rings = False) 
        actions = list(actions)
        state_temp = get_fps(state,length = 2048) 
        if np.random.uniform() > epsilon.value(EPISODE):  
            actions_fps = torch.FloatTensor(np.array(get_fps_list(actions,length = 2048)))
            state_fps = torch.FloatTensor(np.concatenate([state_temp[np.newaxis,:] for i in range(len(actions))]))
            #print(actions.shape,state.shape)
            x = actions_fps + state_fps
            value = self.eval_Q.forward(x)
            top_K_index = np.squeeze(value.detach().numpy()).argsort()[-args.top_k:] #select top_K max Q_value index ; Attention : no need to worry if K were bigger than 'num_actions' 
            top_K_actions = [actions[index] for index in top_K_index]
            action = np.random.choice(top_K_actions) 
            #action = actions[torch.argmax(value)]
        else: 
            action = np.random.choice(actions) 
        action_fps = get_fps(action,length = 2048) 
        return action, state_temp, action_fps 
    def learn(self,data,data_smi):
        if self.step_counter % TARGET_REPLACE_ITER == 0:
            self.target_Q.load_state_dict(self.eval_Q.state_dict())
        self.step_counter += 1
        #data = M.sample(BATCH_SIZE)

        r = torch.FloatTensor(data[:,-1][ :,np.newaxis]) #shape (batch, 1)
        s = torch.FloatTensor(data[:,:args.state_dim]) #shape (batch, args.state_dim)
        a = torch.FloatTensor(data[:,args.state_dim:args.state_dim+args.action_dim]) #shape (batch, args.action_dim)
        s_ = a #shape (batch, args.action_dim)
        
        q_eval = self.eval_Q(s+a) # shape (batch, 1)
        q_next = []
        for batch_index in range(len(data_smi)):
            s_ = data_smi[batch_index]
            s__fps = a[batch_index]
            a_ = getAction(s_,atom_types = ['C','N','O'],
                        allow_removal = True,
                        allow_no_modification = True,
                        allowed_ring_sizes = (3,4,5,6),
                        allow_bonds_between_rings = False)  
            a_fps = torch.FloatTensor(np.array(get_fps_list(a_,length = 2048))) 
            s_expand =  torch.FloatTensor(np.concatenate([s__fps[np.newaxis,:] for i in range(len(a_))])) 
            q_next.append(self.target_Q(a_fps+s_expand).detach()) 
        q_next_max = [i.max() for i in q_next]
        q_target = r + torch.FloatTensor([GAMMA*i for i in q_next_max]).unsqueeze(1) 
        loss = self.loss_function(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class multi_DQN(object):
    def __init__(self,reward_list=['qed'],is_QA=0):
        self.reward_dim = len(reward_list) + is_QA 
        self.step_counter = 0 
        self.loss_function = nn.MSELoss() 
        for i in range(self.reward_dim):
            exec('self.eval_Q_{}=Net()'.format(i))
            exec('self.target_Q_{}=Net()'.format(i))
            exec('self.optimizer_{} = torch.optim.Adam(self.eval_Q_{}.parameters(),lr=args.lr)'.format(i,i))
    def choose_action(self,state): 
        actions = getAction(state,atom_types = ['C','N','O'],
                            allow_removal = True,
                            allow_no_modification = True,
                            allowed_ring_sizes = (3,4,5,6),
                            allow_bonds_between_rings = False) 
        actions = list(actions)
        state_temp = get_fps(state,length = 2048) 
        if np.random.uniform() > epsilon.value(EPISODE): 
            actions_fps = torch.FloatTensor(np.array(get_fps_list(actions,length = 2048)))
            state_fps = torch.FloatTensor(np.concatenate([state_temp[np.newaxis,:] for i in range(len(actions))]))
            x = actions_fps + state_fps
            
            value_list = []
            for i in range(self.reward_dim): 
                exec('value_list.append(self.eval_Q_{}.forward(x))'.format(i))

            value = sum(value_list) 
            top_K_index = np.squeeze(value.detach().numpy()).argsort()[-args.top_k:] 
            top_K_actions = [actions[index] for index in top_K_index]
            action = np.random.choice(top_K_actions)
        else:  
            action = np.random.choice(actions) 
        action_fps = get_fps(action,length = 2048) 
        return action, state_temp, action_fps
    def learn(self,data,data_smi):
        if self.step_counter % TARGET_REPLACE_ITER == 0:
            for i in range(self.reward_dim):
                exec('self.target_Q_{}.load_state_dict(self.eval_Q_{}.state_dict())'.format(i,i))
        self.step_counter += 1

        for i in range(self.reward_dim):
            exec('r_{} = torch.FloatTensor(data[:,{}-self.reward_dim][ :,np.newaxis])'.format(i,i))
        s = torch.FloatTensor(data[:,:args.state_dim]) 
        a = torch.FloatTensor(data[:,args.state_dim:args.state_dim+args.action_dim]) 
        s_ = a
        
        
        for i in range(self.reward_dim):
            exec('q_eval_{} = self.eval_Q_{}(s+a)'.format(i,i))
            exec('q_next_{} = []'.format(i))

        for batch_index in range(len(data_smi)):
            s_ = data_smi[batch_index]
            s__fps = a[batch_index]
            a_ = getAction(s_,atom_types = ['C','N','O'],
                        allow_removal = True,
                        allow_no_modification = True,
                        allowed_ring_sizes = (3,4,5,6),
                        allow_bonds_between_rings = False)
            a_fps = torch.FloatTensor(np.array(get_fps_list(a_,length = 2048)))
            s_expand =  torch.FloatTensor(np.concatenate([s__fps[np.newaxis,:] for i in range(len(a_))])) 
            for i in range(self.reward_dim):
                exec('q_next_{}.append(self.target_Q_{}(a_fps+s_expand).detach())'.format(i,i))

        for i in range(self.reward_dim):
            exec('q_next_max_{} = [i.max() for i in q_next_{}]'.format(i,i))
            exec('q_target_{} = r_{} + torch.FloatTensor([GAMMA*i for i in q_next_max_{}]).unsqueeze(1)'.format(i,i,i))
            exec('loss_{} = self.loss_function(q_eval_{}, q_target_{})'.format(i,i,i))
            exec('self.optimizer_{}.zero_grad()'.format(i))
            exec('loss_{}.backward()'.format(i))
            exec('self.optimizer_{}.step()'.format(i))        
        
if __name__ == "__main__":

    

    env = Properties_ENV_QA(['qed'],init_smi=args.target_mol,max_steps = args.max_step)
    dqn = multi_DQN(['qed'],1)
    M = multi_Memory(args.memory_capacity,reward_dim = dqn.reward_dim)
    M_smi = Memory_smiles(args.memory_capacity)

    file = open(args.output_file,'w')
    
    for EPISODE in range(args.max_episode):
        state = env.reset()
        episode_reward = np.zeros(2) #2: reward_dim
        for STEP in range(args.max_step):
            action,state_fps,action_fps = dqn.choose_action(state)
            state_ , terminate = env.step(action) #take action
            w_ = weight.value(EPISODE) #current weight of reward
            reward = env.reward()
            M.store(state_fps,action_fps,reward) #fps
            M_smi.store(action) #smiles
            
            if M.pointer > args.memory_capacity: 
                data,indexs = M.sample(args.batch_size)
                data_smi = M_smi.sample(indexs)
                dqn.learn(data,data_smi)
            
            state = state_
            episode_reward += np.array(reward) 
            
            if STEP == args.max_step - 1 :
                print('Episode: {}:,Reward: {}'.format(EPISODE,episode_reward))
                #write log
                file.write('EPISODE: '+str(EPISODE)+'\n')    
                for smiles in env.track:
                    file.write(smiles+'\n')
                break
    file.close()
    #torch.save(dqn.eval_Q.state_dict(),'model_state/pretrain...')        
