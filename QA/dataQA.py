#convert moleucle SMILES strings to graph

import torch
import numpy as np
from rdkit import Chem
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

def OneHot(list_,element='None'):
    values = np.array(list_)
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    if element == 'None':
        return onehot_encoded
    elif element in list_:
        index = list_.index(element)
        return onehot_encoded[index].tolist()
    else:
        print('error, wrong feature!',element)
		

Atom_Symbol = ['C','N','O','F','Si','P','S','Cl','Br','I','Na','H','Li','B','K','Mg','Ca','Zn','Ag']
Atom_IsInRing = [True,False]
Atom_Hybridization = ['SP','SP2','SP3','SP3D','SP3D2','S']
#AtomDegree (type:int)
#ImplicitValence (type:int)
Edge_Type = ['SINGLE','DOUBLE','TRIPLE','AROMATIC']


def Mol2Graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    #node features
    AtomSymbol = [atom.GetSymbol() for atom in mol.GetAtoms()]
    ImplicitValence = [atom.GetImplicitValence() for atom in mol.GetAtoms()]
    AtomInRing = [atom.IsInRing() for atom in mol.GetAtoms()]
    AtomDegree = [atom.GetDegree() for atom in mol.GetAtoms()]
    AtomHybridization = [str(atom.GetHybridization()) for atom in mol.GetAtoms()]
    #edge features
    BondType = [str(bond.GetBondType()) for bond in mol.GetBonds()]+[str(bond.GetBondType()) for bond in mol.GetBonds()]
    
    #atom index at begin_site and end_site
    u = torch.tensor([bond.GetBeginAtomIdx() for bond in mol.GetBonds()])
    v = torch.tensor([bond.GetEndAtomIdx() for bond in mol.GetBonds()])
    new_u = torch.cat((u,v))
    new_v = torch.cat((v,u))
    edges = new_u,new_v 
    #build graph
    graph = dgl.graph(edges, num_nodes = len(AtomSymbol), idtype=torch.int32)
    
    #onehot encoding of str features
    AtomSymbol = torch.tensor([OneHot(Atom_Symbol,symbol) for symbol in AtomSymbol])
    AtomInRing = torch.tensor([OneHot(Atom_IsInRing,symbol) for symbol in AtomInRing])
    AtomHybridization = torch.tensor([OneHot(Atom_Hybridization,symbol) for symbol in AtomHybridization])
    BondType = torch.tensor([OneHot(Edge_Type,symbol) for symbol in BondType])
    ImplicitValence = torch.reshape(torch.tensor(ImplicitValence),(len(AtomSymbol),1))
    AtomDegree = torch.reshape(torch.tensor(AtomDegree),(len(AtomSymbol),1))
    
    graph.ndata['AtomSymbol'] = AtomSymbol
    graph.ndata['AtomInRing'] = AtomInRing
    graph.ndata['AtomHybridization'] = AtomHybridization
    graph.ndata['ImplicitValence'] = ImplicitValence
    graph.ndata['AtomDegree'] = AtomDegree
    graph.edata['BondType'] = BondType
	
    graph = dgl.add_self_loop(graph)
    return graph

	
if __name__ == "__main__":
	path = '../data/'
	with open(path+'negative.csv','r') as f:
		n = f.readlines()
	with open(path+'positive.csv','r') as f:
		p = f.readlines()
	labels = torch.cat((torch.ones(len(p),dtype=int),torch.zeros(len(n),dtype=int)),0)
	graphs_p = [Mol2Graph(smiles.strip('\n')) for smiles in p]
	graphs_n = [Mol2Graph(smiles.strip('\n')) for smiles in n]
	graphs = graphs_p + graphs_n 
	
	save_graphs(path+'graph.bin', graphs, {'labels': labels})
	
	
	