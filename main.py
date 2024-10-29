import re
import numpy as np
import torch
import os
import random
from transformers import AutoModelForCausalLM
from stylegan2.makegan import make_stylegan2
from encoder_train import CNNEncoder
import argparse

def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to {seed}")

def get_invariant_terms2save(state_dict, name, selected_tokens):
    dicts = {}
    WqWk_list = []
    WvWo_list = []
    WuWd_list=[]  
    print(state_dict.keys())
    pattern = r'\d+'
    numbers = [int(re.search(pattern, key).group()) for key in state_dict.keys() if re.search(pattern, key)]
    
    # extracted word embedding and last two layers weights
    max_numbers = sorted(numbers, reverse=True)[0]    
    n = [str(max_numbers-1),str(max_numbers)]  
    state_dict = dict(sorted(state_dict.items(), key=lambda x: int(re.search(pattern,x[0]).group()) if re.search(pattern,x[0]) else 0))
    for i in ['model.decoder.embed_tokens.weight','gpt_neox.embed_in.weight','model.embed_tokens.weight','transformer.embedding.word_embeddings.weight','transformer.word_embeddings.weight','word_embeddings.weight','decoder.embed_tokens.weight','transformer.wte.weight','wte.weight']:
        try:
            x=state_dict[i]
            print(i)
            break
        except:
            pass
    x=x[selected_tokens]
    for key, value in state_dict.items():
        l = key.split(".")
        try:
            if l[1] in n or l[2] in n or l[3] in n :
                sub_key = l[-2] + l[-1]
                dicts.setdefault(sub_key, []).append(value)
        except:
            pass

    if name in ["Qwen-7B","Qwen-7B-Chat","MindChat-Qwen-7B","firefly-qwen-7b","Qwen-72B","Qwen-72B-Chat"]:
        for key, value in state_dict.items():
            l = key.split(".")
            try:
                if l[1] in n or l[2] in n or l[3] in n :
                    sub_key = l[-3] + l[-2] + l[-1]
                    dicts.setdefault(sub_key, []).append(value)
            except:
                pass
        for qkv, o_a,o_m, u, d in zip(dicts['attnc_attnweight'],dicts['attnc_projweight'],dicts['mlpc_projweight'],\
                                      dicts['mlpw1weight'],dicts['mlpw2weight']):
            length = qkv.shape[0] // 3
            q = qkv[:length]
            k = qkv[length:2 * length]
            v = qkv[2 * length:]
            WqWk=x@q.t()@k@x.t()
            WqWk_list.append(WqWk)
            WvWo= x@v.t() @ o_a.t()@x.t()
            WvWo_list.append(WvWo)
            WuWd=x@(u.t()*d.t())@o_m.t()@x.t()
            WuWd_list.append(WuWd)
    if name in ["gpt2-large","Cerebras-GPT-1.3B",'geogalactica']:
        dicts={}
        for key, value in state_dict.items():
            l = key.split(".")
            try:
                if l[1] in n or l[2] in n or l[3] in n :
                    sub_key = l[-3] + l[-2] + l[-1]
                    dicts.setdefault(sub_key, []).append(value)
            except:
                pass
        nn=1
        for qkv, o_a,o_m, u in zip(dicts['attnc_attnweight'],dicts['attnc_projweight'],dicts['mlpc_projweight'],\
                                      dicts['mlpc_fcweight']):
            length = qkv.shape[1] // 3
            q = qkv.t()[:length]
            k = qkv.t()[length:2 * length]
            v = qkv.t()[2 * length:]
            WqWk=x@q.t()@k@x.t()
            WqWk_list.append(WqWk)
            WvWo=x@v.t()@o_a.t()@x.t()
            WvWo_list.append(WvWo)
            WuWd=x@o_m.t()@u.t()@x.t()
            WuWd_list.append(WuWd)
            nn=nn+1
    if name in ["chatglm-6b","bloom-7b1","pythia-12b","pythia-6.9b","GPT-NeoXT-Chat-Base-20B","gpt-neox-20b",'bloomz-7b1-mt','bloomz-7b1-p3','oasst-sft-1-pythia-12b','chatglm-fitness-RLHF','pythia-12b-deduped']:
        for qkv, o, u, d in zip(dicts['query_key_valueweight'],dicts['denseweight'],dicts['dense_h_to_4hweight'], 
                                        dicts['dense_4h_to_hweight']):
            length = qkv.shape[0] // 3
            q = qkv[:length]
            k = qkv[length:2 * length]
            v = qkv[2 * length:]
            WqWk=x@q.t()@k@x.t()
            WqWk_list.append(WqWk)
            WvWo=x@v.t()@o.t()@x.t()
            WvWo_list.append(WvWo)
            WuWd=x@u.t()@d.t()@x.t()
            WuWd_list.append(WuWd)
    if name in ["chatglm2-6b","codegeex2-6b"]:
        for qkv, o, u, d in zip(dicts['query_key_valueweight'],dicts['denseweight'],dicts['dense_h_to_4hweight'], 
                                        dicts['dense_4h_to_hweight']):
            len_mini_kv=(qkv.shape[0]-qkv.shape[1])//2
            mini_k = qkv[qkv.shape[1]:qkv.shape[1]+len_mini_kv]
            mini_v = qkv[qkv.shape[1]+len_mini_kv:]
            repeat_times=qkv.shape[1]//len_mini_kv
            q = qkv[:qkv.shape[1]]
            k = torch.cat([mini_k for i in range(repeat_times)],dim=0)
            v = torch.cat([mini_v for i in range(repeat_times)],dim=0)
            g=u[:u.shape[0]//2]
            u=u[u.shape[0]//2:]           
            WqWk=x@q.t()@k@x.t()
            WqWk_list.append(WqWk)
            WvWo=x@v.t()@o.t()@x.t()
            WvWo_list.append(WvWo)
            WuWd=x@u.t()@d.t()@x.t()
            WuWd_list.append(WuWd)
    if name in ["mpt-30b-chat","mpt-30b",'mpt-7b-chat',"mpt-7b-instruct","mpt-7b-storywriter","mpt-30b-instruct","mpt-7b"]:
        for qkv, o, u, d in zip(dicts['Wqkvweight'],dicts['out_projweight'],dicts['up_projweight'], 
                                        dicts['down_projweight']):
            length = qkv.shape[0] // 3
            q = qkv[:length]
            k = qkv[length:2 * length]
            v = qkv[2 * length:]
            WqWk=x@q.t()@k@x.t()
            WqWk_list.append(WqWk)
            WvWo=x@v.t()@o.t()@x.t()
            WvWo_list.append(WvWo)
            WuWd=x@u.t()@d.t()@x.t()
            WuWd_list.append(WuWd)
    if name in ["falcon-40b-instruct","falcon-40b","falcon-40b-sft-top1-560","stablelm-base-alpha-7b","falcon-180B","RedPajama-INCITE-7B-Base",
                'falcon-7b-instruct',
        'falcon-7b','samantha-falcon-7b','falcon-180B-chat','RedPajama-INCITE-7B-Chat','WizardLM-Uncensored-Falcon-7b'
                ]:
        for qkv, o, u, d in zip(dicts['query_key_valueweight'],dicts['denseweight'],dicts['dense_h_to_4hweight'], 
                                        dicts['dense_4h_to_hweight']):
            len_mini_kv=(qkv.shape[0]-qkv.shape[1])//2
            mini_k = qkv[qkv.shape[1]:qkv.shape[1]+len_mini_kv]
            mini_v = qkv[qkv.shape[1]+len_mini_kv:]
            repeat_times=qkv.shape[1]//len_mini_kv
            q = qkv[:qkv.shape[1]]
            k = torch.cat([mini_k for i in range(repeat_times)],dim=0)
            v = torch.cat([mini_v for i in range(repeat_times)],dim=0)
            WqWk=x@q.t()@k@x.t()
            WqWk_list.append(WqWk)
            WvWo=x@v.t()@o.t()@x.t()
            WvWo_list.append(WvWo)
            WuWd=x@u.t()@d.t()@x.t()
            WuWd_list.append(WuWd)
    if name in ["Baichuan-13B-Chat","Baichuan-13B-Base","Baichuan-13B-sft",'Baichuan-7B',
                        'Baichuan-7B-sft',
        'baichuan-7B-chat',
                ]:
        for qkv,o,g,d,u in zip(dicts['W_packweight']\
            ,dicts['o_projweight'],dicts['gate_projweight'],dicts['down_projweight'],dicts['up_projweight']):
            length = qkv.shape[0] // 3
            q = qkv[:length]
            k = qkv[length:2 * length]
            v = qkv[2 * length:]
            WqWk=x@(q.t()@k)@x.t()
            WqWk_list.append(WqWk)
            WvWo=x@(v.t()@o.t())@x.t()
            WvWo_list.append(WvWo)
            WuWd=x@((g.t()*u.t())@d.t())@x.t()
            WuWd_list.append(WuWd)
    if name in ["llama-13b","llama-30b","llama-65b","Guanaco",\
                "internlm-chat-7b","firefly-internlm-7b",
                'Llama-2-7B-fp16',  'BiLLa-7B-SFT', 'internlm-7b', 
                'chinese-alpaca-7b-merged', 'chinese-llama-7b-merged', 'beaver-7b-v1.0', 
                 'Llama-2-7b-chat-fp16', 'medalpaca-7b', 'koala-7b', 'alpaca-native', 
                'vicuna-7b-v1.3', 'alpaca-lora-7b', 'baize-v2-7b', 'wizardLM-7B-HF', 'open_llama_7b',
                  'MiniGPT-4-LLaMA-7B', 'LLaMA-2-7B-32K', 'llama-7b-hf','CodeLlama-7b-hf',
                'Llama2-Chinese-7b-Chat','llama-2-ko-7b', 'Llama-2-7b-chat-hf-function-calling-v2',    
                'Llama2-Chinese-13b-Chat','LLaMA2-13B-Tiefighter','Llama-2-13B-Chat-fp16',
    'Llama-2-13b-hf','Llama-2-13B-fp16-french','LLaMA2-13B-Estopia','Nous-Hermes-Llama2-13b',
    "vicuna-7b-v1.5",'internlm-xcomposer-7b','baichuan-vicuna-7b','Llama-2-7b-WikiChat']:
        for q,k,v,o,g,d,u in zip(dicts['q_projweight'],dicts['k_projweight'],dicts['v_projweight']\
                ,dicts['o_projweight'],dicts['gate_projweight'],dicts['down_projweight'],dicts['up_projweight']):           
            WqWk=x@(q.t()@k)@x.t()
            WqWk_list.append(WqWk)
            WvWo=x@(v.t()@o.t())@x.t()
            WvWo_list.append(WvWo)
            WuWd=x@((g.t()*u.t())@d.t())@x.t()
            WuWd_list.append(WuWd)

    if name in ["opt-30b","galactica-120b","opt-6.7b","galactica-30b",'opt-iml-30b','galactica-30b-evol-instruct-70k']:
        for q,k,v,o,u,d in zip(dicts['q_projweight'],dicts['k_projweight'],dicts['v_projweight']\
                ,dicts['out_projweight'],dicts['fc1weight'],dicts['fc2weight']):
            WqWk=x@q.t()@k@x.t()
            WqWk_list.append(WqWk)
            WvWo=x@v.t()@o.t()@x.t()
            WvWo_list.append(WvWo)
            WuWd=x@u.t()@d.t()@x.t()
            WuWd_list.append(WuWd)
    if name in ["gpt-j-6b"]:
        for q,k,v,o,u,d in zip(dicts['q_projweight'],dicts['k_projweight'],dicts['v_projweight']\
                ,dicts['out_projweight'],dicts['fc_inweight'],dicts['fc_outweight']):
            WqWk=x@q.t()@k@x.t()
            WqWk_list.append(WqWk)
            WvWo=x@v.t()@o.t()@x.t()
            WvWo_list.append(WvWo)
            WuWd=x@u.t()@d.t()@x.t()
            WuWd_list.append(WuWd)
    if name in ["gpt-neo-2.7B"]:
        for q,k,v,o,u,d in zip(dicts['q_projweight'],dicts['k_projweight'],dicts['v_projweight']\
                ,dicts['out_projweight'],dicts['c_fcweight'],dicts['c_projweight']):
            WqWk=x@q.t()@k@x.t()
            WqWk_list.append(WqWk)
            WvWo=x@v.t()@o.t()@x.t()
            WvWo_list.append(WvWo)
            WuWd=x@u.t()@d.t()@x.t()
            WuWd_list.append(WuWd)
    parameters = [torch.stack((t1, t2,t3)) for t1, t2,t3 in zip(WqWk_list, WvWo_list,WuWd_list)]
    parameters = torch.cat(parameters, dim=0)
    np.save(invariant_terms_saved_path+f'{str(name)}.npy', parameters.detach().cpu().numpy())
    return parameters

def mean_pooling(input_tensor):
    # Our target is to perform mean pooling on the input tensor of shape [6, 4096, 4096]
    # such that each [6, 4096, 8] block is reduced to a mean value.
    # We should end up with a tensor of shape [6, 4096, 512] that is
    # then reshaped/flattened to [6*4096*512/4096] = [512].

    # Check if the input tensor has the correct shape
    if input_tensor.shape != (6, 4096, 4096):
        raise ValueError('Input tensor must be of shape [6, 4096, 4096]')
    
    reshaped_tensor = input_tensor.view(512, -1)

    # Perform mean pooling over the last dimension
    pooled_tensor = reshaped_tensor.mean(-1)

    # Flatten the tensor to get a vector of shape [512]
    output_vector = pooled_tensor.view(-1)

    # Normalize the output vector
    mean=torch.mean(output_vector)
    std=torch.std(output_vector)
    output_vector=(output_vector-mean)/std
    return output_vector

def encode(invariant_terms,encoder_path):
    
    CNNencoder = CNNEncoder().cuda()
    encoder_path=encoder_path
    CNNencoder = torch.load(encoder_path)
    CNNencoder.eval()
    #input normalization
    for i in range(invariant_terms.shape[0]):
        mean = torch.mean(invariant_terms[i])
        std = torch.std(invariant_terms[i])
        invariant_terms[i]=(invariant_terms[i]-mean)/std
    output_vector=CNNencoder(invariant_terms.unsqueeze(0).unsqueeze(0).cuda())
    # output normalization
    mean=torch.mean(output_vector)
    std=torch.std(output_vector)
    output_vector=(output_vector-mean)/std
    return output_vector.squeeze(0)

def generate_model_images(feature_vector, png_path, seed):
    seed_everything(seed)
    def convert_to_images(obj):
        """ Convert an output tensor from BigGAN in a list of images.
            Params:
                obj: tensor or numpy array of shape (batch_size, channels, height, width)
            Output:
                list of Pillow Images of size (height, width)
        """
        try:
            import PIL
        except ImportError:
            raise ImportError("Please install Pillow to use images: pip install Pillow")

        if not isinstance(obj, np.ndarray):
            obj = obj.detach().cpu().numpy()

        obj = obj.transpose((0, 2, 3, 1))
        obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)

        img = []
        for i, out in enumerate(obj):
            out_array = np.asarray(np.uint8(out), dtype=np.uint8)
            img.append(PIL.Image.fromarray(out_array))
        return img
    z_matrix=G.G.mapping(feature_vector.unsqueeze(0), None, truncation_psi=G.truncation_psi, truncation_cutoff=G.truncation_cutoff)
    x_n = G(z=z_matrix)  
    img_n=convert_to_images(x_n)
    img_n[0].save(png_path+'.png')
def parse_args():
    parser = argparse.ArgumentParser(description='Encoder Training Script')
    parser.add_argument('--model_path', type=str, default="decapoda-research/llama-7b-hf", help='model path or model name to generate fingerprint')
    parser.add_argument('--huggingface_cache_dir', type=str, default="/data1/byzeng/huggingface/hub", help='your huggingface cache dir')
    parser.add_argument('--sorted_tokens_path', type=str, default="/home/byzeng/project/ICLRcode/toptokenstest/", help='the filefold path of sorted tokens')
    parser.add_argument('--invariant_terms_saved_path', type=str, default="/home/byzeng/NIPS_Code/invariant_terms/", help='the filefold path to save invariant terms')
    parser.add_argument('--fingerprint_saved_path', type=str, default="/home/byzeng/NIPS_Code/fingerprints/", help='the filefold path to save fingerprints')
    parser.add_argument('--encoder_path', type=str, default="/data1/byzeng/goodencodersnew/encoder_n0.4_k48_p1.3_lr0.0001_16_21_10.pth", help='the path of encoder')
    parser.add_argument('--feature_extract_methon', choices=['Mean_pooling', 'CNN'],default='CNN', help='the method to extract feature vector')
    return parser.parse_args() 
if __name__ == "__main__":
   
    # 
    args = parse_args()
    print(args)
    model_path = args.model_path
    huggingface_cache_dir= args.huggingface_cache_dir
    sorted_tokens_path = args.sorted_tokens_path
    invariant_terms_saved_path= args.invariant_terms_saved_path
    fingerprint_saved_path= args.fingerprint_saved_path
    encoder_path= args.encoder_path

    if not os.path.exists(invariant_terms_saved_path):
        os.makedirs(invariant_terms_saved_path)
    if not os.path.exists(fingerprint_saved_path):
        os.makedirs(fingerprint_saved_path)
    num_tokens = 4096
    seed = 100

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path,cache_dir = huggingface_cache_dir,local_files_only=True)
    name = model_path.split("/")[-1]

    # 获取模型的state_dict和selected_tokens
    state_dict = model.state_dict()
    sorted_tokens = []
    with open(sorted_tokens_path+name+'.txt', 'r') as file:
            for line in file:
                sorted_tokens.append(int(line.strip()))
    selected_tokens = sorted_tokens[-num_tokens:]

    # extrct invariant terms from the model
    invariant_terms = get_invariant_terms2save(state_dict, name, selected_tokens)
    if args.feature_extract_methon == 'Mean_pooling':
    #mean pooling then generate fingerprint
        feature_vector=mean_pooling(invariant_terms).cuda()
    else:
    #encode then generate fingerprint
        feature_vector=encode(invariant_terms,encoder_path).cuda()
    
    G=make_stylegan2(model_name='afhqdog').to('cuda')
    generate_model_images(feature_vector,fingerprint_saved_path + name,seed=seed)
    print(f"Model {name} fingerprint generated successfully!")


