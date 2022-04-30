from rdkit import Chem
import numpy as np
from Rdkit_utils import *
import sys 
sys.path.append('..')
from QA.predictQA import *


class molecule_ENV(object):
    def __init__(self,init_smi=None,max_steps=50,target_smi=None,reward_interval=0,early_stop = False):
        self.init_smi = init_smi
        if self.init_smi:
            if not Chem.MolFromSmiles(init_smi):
                raise Exception("Invalid initialization smiles! ",self.init_smi)
            else:
                self.init_mol = Chem.MolFromSmiles(init_smi)
        
        self.max_steps = max_steps
        self.target_smi = target_smi
        self.reward_interval = reward_interval
        self.early_stop = early_stop
        
        self.terminate = False
        self.current_state = init_smi
        self.current_steps = 0
        self.current_action = [] 
        self.track = []
        self.track.append(init_smi)
    def reset(self,scaffold = None): 
        self.current_steps = 0 
        self.terminate = False
        self.track = [] 
        if not scaffold: 
            self.current_state = self.init_smi
        else: 
            if not Chem.MolFromSmiles(scaffold):
                raise Exception("Invalid reset scaffold! ", scaffold)
            else:
                self.current_state = scaffold
        return self.current_state
                
    def get_Properties(self): 
        return Get_Descriptors(self.current_state)
    
    def step(self,action):
        if self.terminate: 
            raise Exception("Terminated episode!")
        else:
            self.current_state = action 
            self.track.append(action) 
            self.current_steps += 1 
            if self.current_steps == self.max_steps: 
                self.terminate = True
        return self.current_state, self.terminate 
    
    def visualization(self,current = True):
        if current: 
            return Chem.Draw.MolToImage(Chem.MolFromSmiles(self.current_state))
        else: 
            mols = [Chem.MolFromSmiles(smi) for smi in self.track if smi]
            return Chem.Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(300,300),legends=['' for x in mols])

parser = argparse.ArgumentParser()
args = parse_arguments(parser)    
model = GIN(
    args.num_layers,args.num_mlp_layers,args.input_dim,args.hidden_dim,args.output_dim,args.final_dropout,args.learn_eps,args.graph_pooling_type,args.neighbor_pooling_type).to('cpu')
model.load_state_dict(torch.load(args.save_path+'model_QA_GIN.pt'))
model.eval()    

with open('../data/positive.csv','r') as f:
    positive = f.readlines()
with open('../data/negative.csv','r') as f:
    negative = f.readlines()    
n,m = len(positive),len(negative)
rank_lib = '../data/rank_library.csv'

            
class Properties_ENV(molecule_ENV):
    def __init__(self,reward_list,init_smi=None,max_steps=50,target_smi=None,reward_interval=0,early_stop = False):
        super(Properties_ENV, self).__init__(init_smi,max_steps,target_smi,reward_interval,early_stop)
        self.reward_list = reward_list 
        self.reward_dim = len(reward_list)
    def reward(self):
        if self.current_state:
            return Get_Descriptors(self.current_state,self.reward_list)
        return [0 for i in self.reward_list]
        
        
        
class Properties_ENV_QA(molecule_ENV):
    def __init__(self,reward_list,init_smi=None,max_steps=50,target_smi=None,reward_interval=0,early_stop = False):
        super(Properties_ENV_QA, self).__init__(init_smi,max_steps,target_smi,reward_interval,early_stop)
        self.reward_list = reward_list 
        self.reward_dim = len(reward_list)+1
    def reward(self):
        if self.current_state:
            MOL = Chem.MolFromSmiles(self.current_state)
            if MOL.GetNumAtoms() < 12: #Given a threshold to avoid trapped
                return Get_Descriptors(self.current_state,self.reward_list)+[0.01]
            return Get_Descriptors(self.current_state,self.reward_list)+[QAscore(n,m,rank_lib,float(predict(model,self.current_state)))]
        return [0 for i in range(self.reward_dim)]


class Properties_ENV_weighted_sum(molecule_ENV):
    def __init__(self,reward_list,init_smi=None,max_steps=50,target_smi=None,reward_interval=0,early_stop = False):
        super(Properties_ENV_weighted_sum, self).__init__(init_smi,max_steps,target_smi,reward_interval,early_stop)
        self.reward_list = reward_list 
        self.reward_dim = len(reward_list)+1 
    def reward(self):
        if self.current_state: 
            MOL = Chem.MolFromSmiles(self.current_state)
            if MOL.GetNumAtoms() < 12: 
                return [np.mean(Get_Descriptors(self.current_state,self.reward_list)+[0.01])]
            return [np.mean(Get_Descriptors(self.current_state,self.reward_list)+[QAscore(n,m,rank_lib,float(predict(model,self.current_state)))])]
        return [0]

class Properties_ENV_singleQA(molecule_ENV):
    def __init__(self,init_smi=None,max_steps=50,target_smi=None,reward_interval=0,early_stop = False):
        super(Properties_ENV_singleQA, self).__init__(init_smi,max_steps,target_smi,reward_interval,early_stop)
    def reward(self):
        if self.current_state: 
            MOL = Chem.MolFromSmiles(self.current_state)
            if MOL.GetNumAtoms() < 12: 
                return 0.01
            return QAscore(n,m,rank_lib,float(predict(model,self.current_state)))
        return 0 


class Properties_ENV_QA_substruct(molecule_ENV):
    def __init__(self,reward_list,init_smi=None,max_steps=50,target_smi=None,reward_interval=0,early_stop = False,target_substruct = ''):
        super(Properties_ENV_QA_substruct, self).__init__(init_smi,max_steps,target_smi,reward_interval,early_stop)
        self.reward_list = reward_list 
        self.reward_dim = len(reward_list)+2 
        self.target_substruct = target_substruct
    def reward(self):
        if self.current_state: 
            MOL = Chem.MolFromSmiles(self.current_state)
            if MOL.GetNumAtoms() < 12:
                return Get_Descriptors(self.current_state,self.reward_list)+[0.01]+[float(substruct_match(self.current_state,self.target_substruct))]
            return Get_Descriptors(self.current_state,self.reward_list)+[QAscore(n,m,rank_lib,float(predict(model,self.current_state)))]+[float(substruct_match(self.current_state,self.target_substruct))]
        return [0 for i in range(self.reward_dim)]






