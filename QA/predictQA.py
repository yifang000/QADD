from .dataQA import *
from .trainQA import *

def predict(model,smiles):
    graph = Mol2Graph(smiles)
    feats = torch.cat((graph.ndata['AtomSymbol'],graph.ndata['AtomInRing'],graph.ndata['AtomHybridization'],graph.ndata['ImplicitValence'],graph.ndata['AtomDegree']),1)
    #return model(graph,feats)
    tensor = model(graph,feats)
    score_tmp = (abs(tensor[0][0])+abs(tensor[0][1]))/2
    score = -score_tmp if tensor[0][1]<0 else score_tmp
    return score

def QAscore(n,m,rank_lib,num):
    #rank-based QAscore
    with open(rank_lib) as f:
        scores = f.readlines()
    scores = [float(i) for i in scores]
    x = len([i for i in scores if i < num])
    return 0.5+(x-m)/(2*n) if x>=m else 0.5+(x-m)/(2*m)
    

if __name__ == '__main__':
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
    
    n_positive,n_negative = len(positive),len(negative)
    rank_lib = '../data/rank_library.csv'
    with open(rank_lib,'w') as f:
        for i in positive+negative:
            f.write(str(float(predict(model,i)))+'\n')
            
    #SMILES = 'C=C1C(=O)OC2C1CC/C(Cn1cc(-c3ccc([N+](=O)[O-])cc3)nn1)=C\CCC1(C)OC21' 
    #print(QAscore(n_positive,n_negative,rank_lib,float(predict(model,SMILES))))
    

    