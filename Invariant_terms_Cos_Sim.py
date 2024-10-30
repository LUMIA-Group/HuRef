import os
import numpy as np
import pandas as pd
import torch
import argparse

def stand_normalize2d(input_matrix):
    normalized_matrix = torch.empty_like(input_matrix)
    for i in range(input_matrix.shape[0]):
        mean = torch.mean(input_matrix[i])
        std = torch.std(input_matrix[i])
        normalized_matrix[i]=(input_matrix[i]-mean)/std
    return normalized_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Encoder Training Script')
    parser.add_argument('--invariant_terms_saved_path', type=str, default="/home/byzeng/project/ICLRcode/invariantstest", help='the filefold path to save invariant terms')
    parser.add_argument('--ics_calcu_models', choices=['llama_family_models', 'offspring_models','independent_models'], default="llama_family_models", help='the models to calculate ICS')
    return parser.parse_args() 

if __name__ == '__main__':
    args = parse_args()
    invariant_terms_saved_path = args.invariant_terms_saved_path
    ics_calcu_models = args.ics_calcu_models
    if ics_calcu_models == 'offspring_models':
        base_models = ['falcon-40b','Llama-2-13b-hf','mpt-30b','Llama-2-7B-fp16','Qwen-7B','Baichuan-13B-Base','internlm-7b']
        offspring_models1 = ['falcon-40b-instruct','Llama-2-13B-Chat-fp16','mpt-30b-chat','Llama-2-7b-chat-fp16','Qwen-7B-Chat','Baichuan-13B-Chat','internlm-chat-7b']
        offspring_models2 = ['falcon-40b-sft-top1-560','Llama-2-13B-fp16-french','mpt-30b-instruct','vicuna-7b-v1.5','firefly-qwen-7b','Baichuan-13B-sft','firefly-internlm-7b']
        for key1,key2,key3 in zip(base_models,offspring_models1,offspring_models2):
            file1_path = os.path.join(invariant_terms_saved_path, f'{key1}.npy')
            file2_path = os.path.join(invariant_terms_saved_path, f'{key2}.npy')
            file3_path = os.path.join(invariant_terms_saved_path, f'{key3}.npy')
            vector1 = torch.from_numpy(np.load(file1_path))
            vector2 = torch.from_numpy(np.load(file2_path))
            vector3 = torch.from_numpy(np.load(file3_path))
            vector1 = stand_normalize2d(vector1)
            vector2 = stand_normalize2d(vector2)
            vector3 = stand_normalize2d(vector3)
            similarity1 = 100*(torch.cosine_similarity(vector1.flatten(), vector2.flatten(), dim=0))
            similarity2 = 100*(torch.cosine_similarity(vector1.flatten(), vector3.flatten(), dim=0))
            print('ICS:',key1,key2,similarity1.item())
            print('ICS:',key1,key3,similarity2.item())
    else:
        if ics_calcu_models == 'independent_models':
            model_names = ['gpt2-large','Cerebras-GPT-1.3B','gpt-neo-2.7B','chatglm-6b','chatglm2-6b','opt-6.7b','pythia-6.9b','llama-7b-hf','Qwen-7B', 'Llama-2-7B-fp16','RedPajama-INCITE-7B-Base', 'bloom-7b1',  'internlm-7b', 
                'open_llama_7b','Baichuan-7B','pythia-12b','llama-13b','gpt-neox-20b',"opt-30b",'llama-30b','galactica-30b','llama-65b',
                'galactica-120b',"falcon-180B"]
        elif ics_calcu_models == 'llama_family_models':
            model_names = [
                'llama-7b-hf','MiniGPT-4-LLaMA-7B','alpaca-native', 'medalpaca-7b','vicuna-7b-v1.3', 'wizardLM-7B-HF','baize-v2-7b','alpaca-lora-7b', 
                'chinese-alpaca-7b-merged','koala-7b', 'chinese-llama-7b-merged','beaver-7b-v1.0',"Guanaco","BiLLa-7B-SFT"
                ]
        cosine_matrix = np.zeros((len(model_names), len(model_names)))
        for i, file1 in enumerate(model_names):
            for j, file2 in enumerate(model_names):
                if i <= j:
                    file1_path = os.path.join(invariant_terms_saved_path, f'{file1}.npy')
                    file2_path = os.path.join(invariant_terms_saved_path, f'{file2}.npy')
                    if os.path.exists(file1_path) and os.path.exists(file2_path):
                        vector1 = torch.from_numpy(np.load(file1_path))
                        vector2 = torch.from_numpy(np.load(file2_path))
                        vector1 = stand_normalize2d(vector1)
                        vector2 = stand_normalize2d(vector2)
                        similarity = 100*(torch.cosine_similarity(vector1.flatten(), vector2.flatten(), dim=0))
                        cosine_matrix[i, j] = similarity
                        cosine_matrix[j, i] = similarity

        # # Create a DataFrame to store cosine similarity
        df = pd.DataFrame(cosine_matrix, columns=model_names, index=model_names)
        cosine_matrix = torch.from_numpy(cosine_matrix)
        lower_triangular = torch.tril(cosine_matrix)

        # Computing mean and variance

        # Mask to exclude the diagonal and upper triangular elements
        mask = torch.tril(torch.ones_like(cosine_matrix), diagonal=-1).bool()
        # Selecting only the lower triangular elements (excluding the diagonal)
        lower_triangular_elements = lower_triangular[mask]
        variance = lower_triangular_elements.std()
        print("ICS std:", variance.item())
        mean = (torch.sum(cosine_matrix)-torch.sum(torch.diag(cosine_matrix))) / (len(model_names)**2-len(model_names))
        print(f'ICS mean={mean}')
        # generate LaTeX table
        latex_table = df.to_latex(caption='ICS values between different models', float_format="%.2f", escape=False)

        # print LaTeX table
        print(latex_table)
