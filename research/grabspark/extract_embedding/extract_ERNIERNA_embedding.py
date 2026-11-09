import os
import pandas as pd
import time
import random
import torch
import numpy as np
from Bio import SeqIO
import sys
try:
    import h5py
except ImportError:
    h5py = None
from pathlib import Path

# Packaged ERNIE-RNA BERT code + checkpoints under GraBSpArk/BERT_new
_BERT_ROOT = Path(__file__).resolve().parents[2] / "BERT_new"
sys.path.insert(0, str(_BERT_ROOT))
from pretrain_code.tasks.mlm import *  # pyright: ignore[reportMissingImports]
from pretrain_code.models.mlm import *  # pyright: ignore[reportMissingImports]
from pretrain_code.tasks.twod_mlm import *  # pyright: ignore[reportMissingImports]
from pretrain_code.criterions.mlm import *  # pyright: ignore[reportMissingImports]
from pretrain_code.models.twod_mlm import *  # pyright: ignore[reportMissingImports]
from pretrain_code.criterions.twod_mlm import *  # pyright: ignore[reportMissingImports]
from load_model import Twod_mlm_onestage, Oned_mlm_onestage, Oned_test, load_model_pretrained, oned_to_twod_bert_input  # pyright: ignore[reportMissingImports]

def read_fasta_biopython(fasta_file):
    """
    使用Biopython读取FASTA文件
    """
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(
            str(record.seq)
        )
    return sequences

def seq_to_rna_index(sequences):
    '''
    input:
    sequences: list of string (difference length)
    
    return:
    rna_index: numpy matrix, shape like: [len(sequences), max_seq_len+2]
    rna_len_lst: list of length
    '''

    rna_len_lst = [len(ss) for ss in sequences]
    max_len = max(rna_len_lst)
    assert max_len <= 1022
    seq_nums = len(rna_len_lst)
    rna_index = np.ones((seq_nums,max_len+2))
    for i in range(seq_nums):
        for j in range(rna_len_lst[i]):
            if sequences[i][j] in set("Aa"):
                rna_index[i][j+1] = 5
            elif sequences[i][j] in set("Cc"):
                rna_index[i][j+1] = 7
            elif sequences[i][j] in set("Gg"):
                rna_index[i][j+1] = 4
            elif sequences[i][j] in set('TUtu'):
                rna_index[i][j+1] = 6
            else:
                rna_index[i][j+1] = 3
        rna_index[i][rna_len_lst[i]+1] = 2 # add 'eos' token
    rna_index[:,0] = 0 # add 'cls' token
    return rna_index, rna_len_lst

def extract_embedding_oned(sequences, if_cls=True, arg_overrides=None, mlm_pretrained_model_path=None, device='cpu'):
    '''
    input:
    sequences: List of string (difference length)
    if_cls: Bool, Determine the size of the extracted feature
    arg_overrides: The folder where the character-to-number mapping file resides
    mlm_pretrained_model_path: The path of the pre-trained model
    device: The driver used by the model
    
    return:
    embedding: numpy matrix, shape like: [len(sequences), 768](if_cls=True) or [len(sequences), max_len_seq+2, 768](if_cls=False)
    '''
    if arg_overrides is None:
        arg_overrides = {"data": str(_BERT_ROOT / "pretrain_dict")}
    if mlm_pretrained_model_path is None:
        mlm_pretrained_model_path = str(_BERT_ROOT / "model" / "mlm_checkpoint" / "checkpoint_best.pt")

    # load model
    model_pretrained = load_model_pretrained(mlm_pretrained_model_path,arg_overrides)
    my_model = Oned_mlm_onestage(model_pretrained.encoder.sentence_encoder).to(device)
    print('Model Loading Done!!!')
    my_model.eval()
    
    # Converts string to rna_index
    rna_index, rna_len_lst = seq_to_rna_index(sequences)
    
    # extract embedding one by one
    if if_cls:
        embedding = np.zeros((len(sequences),768))
    else:
        embedding = np.zeros((len(sequences),max(rna_len_lst)+2,768))
    
    for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
        shorten_index = index[:seq_len+2]
        one_d = torch.from_numpy(shorten_index).long().reshape(1,-1).to(device)
        
        output = my_model(one_d).cpu().detach().numpy()
        
        if if_cls:
            embedding[i] = output[0,0,:]
        else:
            embedding[i, :seq_len+2] = output
    
    return embedding

def extract_embedding_twod(sequences, if_cls=True, arg_overrides=None, mlm_pretrained_model_path=None, device='cpu'):
    '''
    input:
    sequences: List of string (difference length)
    if_cls: Bool, Determine the size of the extracted feature
    arg_overrides: The folder where the character-to-number mapping file resides
    mlm_pretrained_model_path: The path of the pre-trained model
    device: The driver used by the model
    
    return:
    embedding: numpy matrix, shape like: [len(sequences), 768](if_cls=True) or [len(sequences), max_len_seq+2, 768](if_cls=False)
    '''
    
    if h5py is None:
        raise ImportError("h5py is required by extract_embedding_twod, but not by token embedding extraction")
    # load model
    if arg_overrides is None:
        arg_overrides = {"data": str(_BERT_ROOT / "pretrain_dict")}
    if mlm_pretrained_model_path is None:
        mlm_pretrained_model_path = str(_BERT_ROOT / "model" / "two_mlm_checkpoint" / "checkpoint_best.pt")
    model_pretrained = load_model_pretrained(mlm_pretrained_model_path,arg_overrides)
    my_model = Twod_mlm_onestage(model_pretrained.encoder).to(device)
    print('Model Loading Done!!!')
    # print(my_model)
    my_model.eval()
    
    # Converts string to rna_index
    rna_index, rna_len_lst = seq_to_rna_index(sequences)
    
    begin_time = time.time()
    suo_list = list(range(5,23845))
    random.shuffle(suo_list)
    for j in suo_list[:1000]:
        sequences = read_fasta_biopython(f'/data1/zshuo23/GraBSpArk/data/RNA/Similarity_deduplication/shard/shard_{j}.fasta')
        # seqs = ['GAGGAAAGTCCGGGCTCCACAGGGCAGGGTGGTTGCTAACGGCAACCCGGGGTGACCCGCGGGACAGTGCCACAGAAAACAGACCGCCCGGCCGACAGCGGCCGGGTAAGGGTGAAACGGTGGTGTAAGAGACCACCAGCGCCTCGGGTGACCGGGGCGGCTCGGTAAACCCCACCCGGAGCAAGGCCAAGAGGAAGCCCTCGGGCTTCTGCGCAGGCGTTCGAGGGCGGCCCGTCCGATGCCCGCGGGTAGGCCGCTAGAGCCTGTCGGCGACGGCAGGCCCAGATGGATGGTCGCCCATCGTCCGCCGCGAGGCGGGCGTGGATAGAACCCGGCGTAG']
        print(len(sequences))
        # extract embedding one by one
        if if_cls:
            embedding = np.zeros((len(sequences),768))
        else:
            embedding = np.zeros((len(sequences),1022+2,768))

        # for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
        for i in range(len(sequences)):
            index,seq_len = seq_to_rna_index([sequences[i]])
            one_d, two_d = oned_to_twod_bert_input(index[0],1022)
            one_d = one_d.to(device)
            two_d = two_d.to(device)
            
            output, logits = my_model(one_d,two_d)
            output = output.cpu().detach().numpy()
            logits = logits.cpu().detach().numpy()
            
            if if_cls:
                embedding[i] = output[0,0,:]
            else:
                embedding[i, :1022+2] = output
            # print(i)
        print("Total time for one shard:", time.time() - begin_time)
        with h5py.File(f'/data1/zshuo23/GraBSpArk/data/RNA/Similarity_deduplication/ERNIERNA_embeddings/layer_12/shard_{j}.h5', "w") as f:
            f.create_dataset(
                "data",
                data=embedding,
                compression="gzip",
                compression_opts=9      # 1–9，越大越慢但越省空间
            )
        print("Total time for one shard:", time.time() - begin_time)
    return embedding

def extract_embedding_twod_token(sequences, if_cls=True, arg_overrides=None, mlm_pretrained_model_path=None, device='cpu'):
    '''
    input:
    sequences: List of string (difference length)
    if_cls: Bool, Determine the size of the extracted feature
    arg_overrides: The folder where the character-to-number mapping file resides
    mlm_pretrained_model_path: The path of the pre-trained model
    device: The driver used by the model
    
    return:
    embedding: numpy matrix, shape like: [len(sequences), 768](if_cls=True) or [len(sequences), max_len_seq+2, 768](if_cls=False)
    '''
    
    # load model
    if arg_overrides is None:
        arg_overrides = {"data": str(_BERT_ROOT / "pretrain_dict")}
    if mlm_pretrained_model_path is None:
        mlm_pretrained_model_path = str(_BERT_ROOT / "model" / "two_mlm_checkpoint" / "checkpoint_best.pt")
    model_pretrained = load_model_pretrained(mlm_pretrained_model_path,arg_overrides)
    my_model = Twod_mlm_onestage(model_pretrained.encoder).to(device)
    print('Model Loading Done!!!')
    # print(my_model)
    my_model.eval()
    
    # Converts string to rna_index
    arrays = [] # 先收集

    rna_index, rna_len_lst = seq_to_rna_index(sequences)
    
    for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
        
        one_d, two_d = oned_to_twod_bert_input(index,seq_len)
        one_d = one_d.to(device)
        two_d = two_d.to(device)
        
        output = my_model(one_d,two_d).cpu().detach().numpy()
        # print(output.shape, seq_len)
        if i % 2000 == 0:
            print(i)
        arrays.append(output[0, 1:-1, :])
        # print(i)
    #         for arr in array_list:
    # arrays.append(arr)
    result = np.concatenate(arrays, axis=0)
    # print(result.shape)
    return result

def extract_embedding_twod_attnmap(sequences, attn_len=None, arg_overrides=None, mlm_pretrained_model_path=None, device='cpu'):
    '''
    input:
    sequences: List of string (difference length)
    attn_len: Int (Complement the sequence to this length). if attn_len=None, atten_len will be the length of the longest sequence in the sequences
    arg_overrides: The folder where the character-to-number mapping file resides
    mlm_pretrained_model_path: The path of the pre-trained model
    device: The driver used by the model
    
    return:
    atten_map: numpy matrix, shape like: [len(sequences), attn_len+2, attn_len+2]
    '''
    
    # load model
    if arg_overrides is None:
        arg_overrides = {"data": str(_BERT_ROOT / "pretrain_dict")}
    if mlm_pretrained_model_path is None:
        mlm_pretrained_model_path = str(_BERT_ROOT / "model" / "two_mlm_checkpoint" / "checkpoint_best.pt")
    model_pretrained = load_model_pretrained(mlm_pretrained_model_path,arg_overrides)
    my_model = Twod_mlm_onestage(model_pretrained.encoder).to(device)
    print('Model Loading Done!!!')
    my_model.eval()
    
    # Converts string to rna_index
    rna_index, rna_len_lst = seq_to_rna_index(sequences)
    
    # extract embedding one by one
    if attn_len == None:
        attn_len = max(rna_len_lst)

    rna_attn_map_embedding = np.zeros((len(sequences),(attn_len+2), (attn_len+2)))

    for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
        one_d, two_d = oned_to_twod_bert_input(index,seq_len)
        one_d = one_d.to(device)
        two_d = two_d.to(device)
        
        output = my_model(one_d,two_d,return_attn_map=True).cpu().detach().numpy()
        
        rna_attn_map_embedding[i, :(seq_len+2), :(seq_len+2)] = output
        
    return rna_attn_map_embedding

def extract_embedding_onehot(sequences,attn_len=None):
    '''
    input:
    sequences: list of string (difference length)
    attn_len: int (Complement the sequence to this length). if attn_len=None, atten_len will be the length of the longest sequence in the sequences
    
    return:
    embedding: numpy matrix, shape like: [len(sequences), 4*attn_len]
    '''
    
    rna_len_lst = [len(seq) for seq in sequences]
    max_len = max(rna_len_lst)
    assert max_len <= 1022
    seq_nums = len(rna_len_lst)
    if attn_len == None:
        attn_len = max_len
    rna_index = np.zeros((seq_nums,4*attn_len))
    for i in range(seq_nums):
        for j in range(rna_len_lst[i]):
            if sequences[i][j] in set("Aa"):
                rna_index[i][j*4:(j+1)*4] = [1,0,0,0]
            elif sequences[i][j] in set("Cc"):
                rna_index[i][j*4:(j+1)*4] = [0,1,0,0]
            elif sequences[i][j] in set("Gg"):
                rna_index[i][j*4:(j+1)*4] = [0,0,1,0]
            elif sequences[i][j] in set('TUtu'):
                rna_index[i][j*4:(j+1)*4] = [0,0,0,1]
            else:
                rna_index[i][j*4:(j+1)*4] = [0,0,0,0]
    return rna_index


def extract_test_oned(sequences, if_cls=True, arg_overrides=None, mlm_pretrained_model_path=None, device='cpu'):
    '''
    input:
    sequences: List of string (difference length)
    if_cls: Bool, Determine the size of the extracted feature
    arg_overrides: The folder where the character-to-number mapping file resides
    mlm_pretrained_model_path: The path of the pre-trained model
    device: The driver used by the model
    
    return:
    embedding: numpy matrix, shape like: [len(sequences), 768](if_cls=True) or [len(sequences), max_len_seq+2, 768](if_cls=False)
    '''
    
    # load model
    if arg_overrides is None:
        arg_overrides = {"data": str(_BERT_ROOT / "pretrain_dict")}
    if mlm_pretrained_model_path is None:
        mlm_pretrained_model_path = str(_BERT_ROOT / "model" / "mlm_checkpoint" / "checkpoint_best.pt")
    model_pretrained = load_model_pretrained(mlm_pretrained_model_path,arg_overrides)
    my_model = Oned_test(model_pretrained.encoder).to(device)
    print('Model Loading Done!!!')
    total_params = sum(p.numel() for p in my_model.parameters())

    # 将参数量转换为百万(M)单位
    total_params_in_million = total_params / 1_000_000

    print(f"模型的参数总量: {total_params}")
    print(f"模型的参数量 (百万): {total_params_in_million} M")
    my_model.eval()
    
    # Converts string to rna_index
    rna_index, rna_len_lst = seq_to_rna_index(sequences)
    
    # extract embedding one by one
    if if_cls:
        embedding = np.zeros((len(sequences),768))
    else:
        embedding = np.zeros((len(sequences),max(rna_len_lst)+2,768))
    
    for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
        shorten_index = index[:seq_len+2]
        one_d = torch.from_numpy(shorten_index).long().reshape(1,-1).to(device)
        
        emb, logit = my_model(one_d)
        emb = emb.cpu().detach().numpy()
        logit = logit.cpu().detach().numpy()
        
        if if_cls:
            embedding[i] = emb[0,0,:]
        else:
            embedding[i, :seq_len+2] = emb
    
        print(emb.shape, logit[0].shape)
        print(rna_index[0])
        print(np.argmax(logit[0, 1:-1, 0:8], axis=1))
        
    return embedding, logit

import torch
import torch.nn as nn
import torch.nn.functional as F


class ERNIERNALogitsHead(nn.Module):
    def __init__(
        self,
        layer_norm=None,
        activation_fn=None,
        lm_head_transform_weight=None,
        embed_tokens_weight=None,
        embed_out=None,
        lm_output_learned_bias=None,
        share_input_output_embed=True,
    ):
        super().__init__()

        # 直接引用原模型的模块与参数，不复制，不改变
        if layer_norm is not None:
            self.layer_norm = layer_norm
        else:
            self.layer_norm = nn.LayerNorm(768)

        if lm_head_transform_weight is not None:
            self.lm_head_transform_weight = lm_head_transform_weight
        else:
            self.lm_head_transform_weight = nn.Linear(768, 768)

        if embed_tokens_weight is not None:
            self.embed_tokens_weight = embed_tokens_weight
        else:
            self.embed_tokens_weight = nn.Parameter(torch.empty(25, 768))
        
        if lm_output_learned_bias is not None:
            self.lm_output_learned_bias = lm_output_learned_bias
        else:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(25))

        if activation_fn is not None:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = F.gelu
        self.embed_out = embed_out
        self.share_input_output_embed = share_input_output_embed


    def forward(self, x, masked_tokens=None):
        """
        x: (B, T, C)
        """

        if masked_tokens is not None:
            x = x[masked_tokens, :]

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        if self.share_input_output_embed and self.embed_tokens_weight is not None:
            x = F.linear(x, self.embed_tokens_weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)

        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        return x

if __name__ == "__main__":
    
    current_path = str(_BERT_ROOT)
    
    seqs = ['GAAATTATCTGCAGGAAATGTGAA']
    arg_overrides = { "data": current_path + '/pretrain_dict/' }
    mlm_pretrained_model_path =  current_path + '/model/mlm_checkpoint/checkpoint_best.pt'
    
    two_mlm_pretrained_model_path =  current_path + '/model/two_mlm_checkpoint/checkpoint_best.pt'
    device='cuda:5'
    # device='cpu'
    begin_time = time.time()
    # model_pretrained = load_model_pretrained(two_mlm_pretrained_model_path,arg_overrides)
    # my_model = Twod_mlm_onestage(model_pretrained.encoder)

    # output_head = ERNIERNALogitsHead(
    #     layer_norm=my_model.sentence_encoder.layer_norm,
    #     activation_fn=my_model.sentence_encoder.activation_fn,
    #     lm_head_transform_weight=my_model.sentence_encoder.lm_head_transform_weight,
    #     embed_tokens_weight=getattr(my_model.sentence_encoder.sentence_encoder.embed_tokens, "weight", None),
    #     embed_out=my_model.sentence_encoder.embed_out,
    #     lm_output_learned_bias=my_model.sentence_encoder.lm_output_learned_bias,
    #     share_input_output_embed=my_model.sentence_encoder.share_input_output_embed,
    # )
    # print(output_head.activation_fn)
    # print(output_head.embed_out)
    # print(output_head.share_input_output_embed)
    # torch.save(output_head.state_dict(), "/data1/zshuo23/GraBSpArk/ckpt/ERNIERNA_LogitsHead/model_weights.pt")

    # output_head = ERNIERNALogitsHead()
    # output_head.load_state_dict(torch.load("/data1/zshuo23/GraBSpArk/ckpt/ERNIERNA_LogitsHead/model_weights.pt"))
    # output_head.eval()
    
    # extract_test_oned(seqs, if_cls=True, arg_overrides=arg_overrides, mlm_pretrained_model_path=mlm_pretrained_model_path, device=device)
    
    # cls_embedding = extract_embedding_oned(seqs, if_cls=True, arg_overrides=arg_overrides, mlm_pretrained_model_path=mlm_pretrained_model_path, device=device)
    # print(cls_embedding.shape) # cls_embedding shape like [3, 768]
    
    # all_embedding = extract_embedding_oned(seqs, if_cls=False, arg_overrides=arg_overrides, mlm_pretrained_model_path=mlm_pretrained_model_path, device=device)
    # print(all_embedding.shape) # all_embedding shape like [3, 10, 768]
    # seqs = read_fasta_biopython('/data1/zshuo23/GraBSpArk/data/RNA/rnacentral_v25_clustered_rep_seq_min-seq-id_0.8_c_0.6_max1022/val_8_1_1.fasta')
    # seqs = read_fasta_biopython('/data/zry20/RNAstructure/ERNIE-RNA/ss-downstream/rivas/data/TestSetA_removeX.fasta')
    # df = pd.read_csv('/data1/zshuo23/GraBSpArk/data/RNA/RNASSTR/ERNIERNA_embeddings/RNASSTR_len401-1022_train.csv')
    # seqs = df['sequence'].tolist()
    # seqs = ['GAGGAAAGTCCGGGCTCCACAGGGCAGGGTGGTTGCTAACGGCAACCCGGGGTGACCCGCGGGACAGTGCCACAGAAAACAGACCGCCCGGCCGACAGCGGCCGGGTAAGGGTGAAACGGTGGTGTAAGAGACCACCAGCGCCTCGGGTGACCGGGGCGGCTCGGTAAACCCCACCCGGAGCAAGGCCAAGAGGAAGCCCTCGGGCTTCTGCGCAGGCGTTCGAGGGCGGCCCGTCCGATGCCCGCGGGTAGGCCGCTAGAGCCTGTCGGCGACGGCAGGCCCAGATGGATGGTCGCCCATCGTCCGCCGCGAGGCGGGCGTGGATAGAACCCGGCGTAG']
    # begin_index = 0
    # end_index = len(seqs)
    # print(len(seqs[begin_index:end_index]))
    # token_embedding = extract_embedding_twod_token(seqs[begin_index:end_index], if_cls=True, arg_overrides=arg_overrides, mlm_pretrained_model_path=two_mlm_pretrained_model_path, device=device)

    # np.save(f'/data1/zshuo23/GraBSpArk/data/RNA/RNASSTR/ERNIERNA_embeddings/layer_12/RNASSTR_len401-1022_train_token_embedding.npy', token_embedding)

    # cls_embedding = extract_embedding_twod(seqs, if_cls=True, arg_overrides=arg_overrides, mlm_pretrained_model_path=two_mlm_pretrained_model_path, device=device)
    # print(cls_embedding.shape) # cls_embedding shape like [3, 768]

    # all_embedding, logits = extract_embedding_twod(seqs, if_cls=False, arg_overrides=arg_overrides, mlm_pretrained_model_path=two_mlm_pretrained_model_path, device=device)
    # print(all_embedding.shape, logits.shape) # cls_embedding shape like [3, 768]
    # recover_logits = output_head(torch.tensor(all_embedding, dtype=torch.float32)).cpu().detach().numpy()
    # print(recover_logits.shape) # recover_logits shape like [3, 10, 25]
    # print(np.mean((recover_logits - logits)**2))


    # all_embedding = extract_embedding_twod(seqs, if_cls=False, arg_overrides=arg_overrides, mlm_pretrained_model_path=two_mlm_pretrained_model_path, device=device)


        # np.save(f'/data1/zshuo23/GraBSpArk/data/RNA/RDL1.6.1-dataset2-seqs-subset10000/ERNIERNA_embeddings/layer_12/shard_{i}.npy', all_embedding)
    
    seqs = ['GCGCCCUUAGCUCAGUUGGAUAGAGCAACGGCCUUCUAAGCCGUGGGUCGCAGGUUCGAAUCCUGCAGGGCGCG', 'GCGCCCGUAGCUCAGUUGGAUAGAGCAACGGCCUUCUAAGCCGUGGGUCGCAGGUUCGAAUCCUGCCGGGCGCG', 'GCGCCCGUAGCUCAGCUGGAUAGAGCAUCGGCCUUCUAAGCCGGGGGUCGCAGGUUCGAAUCCUGCCGGGCGCG', 'GCGCCCAUAGCUCAGCUGGAUAGAGCAACGGCCUUCUAAGCCGUGGGUCGCAGGUUCGAAUCCUGCUGGGCGCG']

    twod_attnmap = extract_embedding_twod_attnmap(seqs, attn_len=None, arg_overrides=arg_overrides, mlm_pretrained_model_path=two_mlm_pretrained_model_path, device=device)
    print(twod_attnmap.shape) # twod_attnmap shape like [3, 10, 10]
    np.save('/data1/zshuo23/SEGDPO/tests/twod_attnmap.npy', twod_attnmap)

    # onehot_embedding = extract_embedding_onehot(seqs)
    # print(onehot_embedding.shape) # onehot_embedding shape like [3, 4*8]
    
    print("Total time:", time.time() - begin_time)

    # nohup python -u extract_embedding_ERNIERNA.py > /data1/zshuo23/GraBSpArk/data/RNA/rnacentral_v25_clustered_rep_seq_min-seq-id_0.8_c_0.6_max1022/ERNIERNA_embeddings/output.out 2>&1 &
    # 857531
