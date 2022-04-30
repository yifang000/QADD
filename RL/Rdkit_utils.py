import random
from rdkit.Chem import QED
from rdkit import Chem 
from rdkit.Chem import RDConfig 
from rdkit.Chem import Descriptors
import os 
import sys 
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Recap
from rdkit.Chem import BRICS
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs
import numpy as np

def Get_Descriptors(smi,reward_list) -> list:
    '''
    Input:  smiles
    Output: molecular properties
    '''
    mol = Chem.MolFromSmiles(smi)
    mol_properties = []
    if 'qed' in reward_list:
        mol_properties.append(QED.qed(mol))
    if 'sascore' in reward_list:
        sascore = sascorer.calculateScore(mol)
        norm_sascore = -1/9*(sascore-10)
        mol_properties.append(norm_sascore)
    if 'molwt' in reward_list: 
        mol_properties.append(1-abs(Descriptors.MolWt(mol)-500)/500) #500: target molwt
    if 'logp' in reward_list:
        mol_properties.append(Descriptors.MolLogP(mol))
    if 'tpsa' in reward_list:
        mol_properties.append(Descriptors.TPSA(mol))
    return mol_properties


def Add_Element_Single(mol,atoms_free,element) -> list:
    '''
    Input: -mol: rdkit.Chem.rdchem.Mol 
           -atoms_free: 含有空余键位的原子序号
           -element: str 待添加的卤素
    Output: list(smiles)
    '''
    
    Atom_Add = []
    for atom in atoms_free:
        new_mol = Chem.RWMol(mol)
        index = new_mol.AddAtom(Chem.Atom(element))
        new_mol.AddBond(atom, index, Chem.BondType.SINGLE)
        sanitization_result = Chem.SanitizeMol(new_mol, catchErrors=True)
        if sanitization_result:
            continue
        Atom_Add.append(Chem.MolToSmiles(new_mol))  
    return list(set(Atom_Add))

def Add_FunctionGroup_Single(smi) -> list:
    '''
    Input: smiles
    Output: list(smiles) 
    '''
    mol = Chem.MolFromSmiles(smi)
    atoms_free = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumImplicitHs() >= 1]
    Atom_Add = Add_Element_Single(mol,atoms_free,'Br')
    Atom_Add_FG = []
    #function groups
    FGs = ['N=C=S','S(N)(=O)=O','NS(C)(=O)=O','S(=O)(=O)O','S(=O)(=O)OC','S(C)(=O)=O','S(=O)(=O)Cl','S(C)=O','SC','S','S=C',
          'Cl','F','C(F)(F)F','Br'
    ]
    
    for smiles in Atom_Add:
        for functiongroup in FGs:
            mol = Chem.MolFromSmiles(smiles)
            pattern = Chem.MolFromSmiles('Br')
            replace = Chem.MolFromSmiles(functiongroup)
            rms = AllChem.ReplaceSubstructs(mol, pattern, replace)
            rms_new = [mol for mol in rms if not Chem.SanitizeMol(mol, catchErrors=True)]
            for i in rms_new:
                Atom_Add_FG.append(Chem.MolToSmiles(i))
    return list(set(Atom_Add_FG+Atom_Add))

'''
#example:
Atom_Add_FG_single = Add_FunctionGroup_Single('Cc1ccccc1') 
#visualization
ms = [Chem.MolFromSmiles(smi) for smi in Atom_Add_FG_single]
Chem.Draw.MolsToGridImage(ms,molsPerRow=4,subImgSize=(300,300),legends=['' for x in ms])
'''
def Add_FunctionGroup_Double(smi, random = False) -> list:
    '''
    Input: smiles
           random = False/True
    Output: list(smiles) 
    '''    
    mol = Chem.MolFromSmiles(smi)
    atoms_free = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumImplicitHs() >= 1]
    Atom_Add = Add_Element_Single(mol,atoms_free,'Br')
    Atom_Add_2 = []
    for smi in Atom_Add:
        mol = Chem.MolFromSmiles(smi)
        atoms_free = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumImplicitHs() >= 1]
        Atom_Add_ = Add_Element_Single(mol,atoms_free,'I')
        Atom_Add_2 += Atom_Add_
    
    Atom_Add_FG = []
    FGs = ['N=C=S','S(N)(=O)=O','NS(C)(=O)=O','S(=O)(=O)O','S(=O)(=O)OC','S(C)(=O)=O','S(=O)(=O)Cl','S(C)=O','SC','S','S=C',
          'F','C(F)(F)F','Cl','Br'
    ]
    
    for smiles in Atom_Add_2:
        mol = Chem.MolFromSmiles(smiles)
        pattern1 = Chem.MolFromSmiles('Br')
        pattern2 = Chem.MolFromSmiles('I')
        if random: 
            functiongroup1,functiongroup2 = random.sample(FGs,2)
            replace1 = Chem.MolFromSmiles(functiongroup1)
            replace2 = Chem.MolFromSmiles(functiongroup2)
            rms = AllChem.ReplaceSubstructs(mol, pattern1, replace1)
            rms2 = AllChem.ReplaceSubstructs(rms[0], pattern2, replace2)
            rms_new = [mol for mol in rms2 if not Chem.SanitizeMol(mol, catchErrors=True)]
            for i in rms_new:
                Atom_Add_FG.append(Chem.MolToSmiles(i))
        else:
            for functiongroup1 in FGs:
                for functiongroup2 in FGs:
                    replace1 = Chem.MolFromSmiles(functiongroup1)
                    replace2 = Chem.MolFromSmiles(functiongroup2)
                    rms = AllChem.ReplaceSubstructs(mol, pattern1, replace1)
                    rms2 = AllChem.ReplaceSubstructs(rms[0], pattern2, replace2)
                    rms_new = [mol for mol in rms2 if not Chem.SanitizeMol(mol, catchErrors=True)]
                    for i in rms_new:
                        Atom_Add_FG.append(Chem.MolToSmiles(i))
    return list(set(Atom_Add_FG))

#remove chirality
def Remove_H(smi) -> str:
    return smi.replace('[C@@H]','C').replace('[C@H]','C').replace('[C@@]','C').replace('[C@]','C')

def Get_Scaffold(smi) -> str:
    '''
    Murcko scaffold
    Input : smiles of mol
    Output: smiles of mol Murcko Scaffold
    '''
    mol = Chem.MolFromSmiles(smi)
    core_mol= MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core_mol)

def Get_Split_Mol_Recap(smi) -> list:
    '''
    Recap
    Lewell X Q et al. 
    Recap retrosynthetic combinatorial analysis procedure: a powerful new technique for identifying privileged molecular fragments with useful applications in combinatorial chemistry[J].
    Journal of chemical information and computer sciences, 1998, 38(3): 511-522.
    '''
    mol = Chem.MolFromSmiles(smi)
    hierarch = Recap.RecapDecompose(mol)
    leaves = list(hierarch.GetLeaves().values())
    mol_split = [X for X in [Chem.MolToSmiles(x.mol) for x in leaves]]
    return mol_split

def Get_Split_Mol_BRICS(smi) -> list:
    '''
    BRICS
    Degen J et al. 
    On the Art of Compiling and Using 'Drug‐Like’ Chemical Fragment Spaces[J].
    ChemMedChem: Chemistry Enabling Drug Discovery, 2008, 3(10): 1503-1507.
    '''
    m = Chem.MolFromSmiles(smi)
    frags = [BRICS.BRICSDecompose(m)]
    return frags

def Butina_ClusterFps(smi_list,cutoff) -> tuple:
    '''
    Input: list of smiles 
           cutoff : threshold（default 0.2）
    Output:tuple of cluster(tuple): (cluster1,cluster2...)
    '''
    mols = [Chem.MolFromSmiles(smi) for smi in smi_list]
    mols = [mol for mol in mols if mol is not None]
    #morgan fingerprints
    footprints = [AllChem.GetMorganFingerprintAsBitVect(i,2,1024) for i in mols]
    distance_matrix = []
    length = len(footprints)
    for i in range(1,length):
        sim = DataStructs.BulkTanimotoSimilarity(footprints[i],footprints[:i])
        distance_matrix.extend([1-x for x in sim])
    clusters = Butina.ClusterData(distance_matrix,length,cutoff,isDistData=True)
    return clusters

def get_fps(smiles, length):
    if smiles is None:
        return np.zeros((length,))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros((length,))
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 3, length)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr

def get_fps_list(smiles_list,length):
    return [get_fps(i,length) for i in smiles_list]

def substruct_match(smiles,target):
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmiles(target)
    return mol.HasSubstructMatch(pattern)
