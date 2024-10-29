import torch
from transformers import AutoModelForCausalLM
def model_cosine_similarity(model1,model2path,huggingface_cache_dir):
    model2 = AutoModelForCausalLM.from_pretrained(model2path, cache_dir=huggingface_cache_dir,local_files_only=True,trust_remote_code=True)
    parameters1=list(model1.parameters())
    parameters2=list(model2.parameters())
    dot_list = []
    l2normi_list = []
    l2normj_list = []
    for i, j in zip(parameters1, parameters2):
        flat_i = torch.flatten(i)
        flat_j = torch.flatten(j)
        try:
            dot_list.append(torch.dot(flat_i, flat_j))
            l2normi_list.append(torch.sum(flat_i*flat_i) )
            l2normj_list.append(torch.sum(flat_j*flat_j) )
        #ignore the mismatched parameters, for example, the mismatched embedding parameters
        except:
            pass
    model_cos=torch.sum(torch.stack(dot_list))/torch.sqrt(torch.sum(torch.stack(l2normi_list)))/torch.sqrt(torch.sum(torch.stack(l2normj_list)))
    return model_cos
if __name__ == '__main__':
    llamapath = 'decapoda-research/llama-7b-hf'
    huggingface_cache_dir = '/data1/byzeng/huggingface/hub'
    llama = AutoModelForCausalLM.from_pretrained(llamapath, cache_dir=huggingface_cache_dir,local_files_only=True,trust_remote_code=True)
    # model2path = "minlik/chinese-alpaca-7b-merged"
    # pathlist=[ 
    #         # "TheBloke/Llama-2-7B-fp16","JosephusCheung/Guanaco","Neutralzz/BiLLa-7B-SFT",
    #         # "internlm/internlm-7b","minlik/chinese-alpaca-7b-merged",
    #         # "/data1/byzeng/huggingface/hub/models--minlik--chinese-llama-7b-merged/snapshots/1ca4d87576f1fef4d44a949fb65bbe6b96675872",
    #         # 'PKU-Alignment/beaver-7b-v1.0',
    #         # "baichuan-inc/baichuan-7B",
    #         # "medalpaca/medalpaca-7b",
    #         "samwit/koala-7b","chavinlo/alpaca-native","lmsys/vicuna-7b-v1.3","chainyo/alpaca-lora-7b","project-baize/baize-v2-7b",
    #         "TheBloke/wizardLM-7B-HF","openlm-research/open_llama_7b","wangrongsheng/MiniGPT-4-LLaMA-7B"]
    pathlist=[ 
        #  "TheBloke/Llama-2-7B-fp16",
        # "internlm/internlm-7b","minlik/chinese-alpaca-7b-merged",
        # "/data1/byzeng/huggingface/hub/models--minlik--chinese-llama-7b-merged/snapshots/1ca4d87576f1fef4d44a949fb65bbe6b96675872",
        "baichuan-inc/baichuan-7B",
        "medalpaca/medalpaca-7b",
        "samwit/koala-7b","chavinlo/alpaca-native","lmsys/vicuna-7b-v1.3","chainyo/alpaca-lora-7b","project-baize/baize-v2-7b",
        "TheBloke/wizardLM-7B-HF","openlm-research/open_llama_7b","wangrongsheng/MiniGPT-4-LLaMA-7B"]
    for model2path in pathlist:
        model_cos = model_cosine_similarity(llama,model2path,huggingface_cache_dir)
        print(f'PCS between llama-7b and {model2path}',model_cos.item())
