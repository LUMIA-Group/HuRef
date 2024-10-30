import collections
from tqdm import tqdm
import concurrent.futures
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer
import os

def process_chunk(tokenizer,chunk):
    word_freq = collections.Counter()
    for item in chunk:
        # Count word frequency
        tokens = tokenizer.tokenize(item)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)        
        # Update word frequency count
        word_freq.update(token_ids)
    return word_freq

def process(data, num_threads,tokenizer):
    result = collections.Counter()
    chunk_size = len(data) // num_threads
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_chunk,tokenizer,chunk) for chunk in chunks]
        for future in tqdm(concurrent.futures.as_completed(futures)):
            result += future.result()

    return result

def filter_nonenglish_text(example):
    return all((ord(char) < 592 or (ord(char) in range(1024,1279)))  for char in example)#remain Latin and Cyrillic

def load_tokenizer(path):
    for tokenizer_class in [LlamaTokenizer, AutoTokenizer]:
        try:
            return tokenizer_class.from_pretrained(
                path,
                trust_remote_code=True,
                unk_token="<unk>",
                bos_token="<s>",
                eos_token="</s>"
            )
        except:
            try:
                return tokenizer_class.from_pretrained(path, trust_remote_code=True)
            except:
                continue
    raise ValueError(f"Failed to load tokenizer from {path}")

def sort_tokens_frequency(path,savepath,datanum=400000,num_process=40):
    tokenizer = load_tokenizer(path)
    dataset=load_dataset("wikipedia", "20220301.en",cache_dir="/data1/byzeng/huggingface/datasets")
    word_freq = collections.Counter()
    for i in range(len(dataset["train"]["text"]) // datanum):
        data = dataset["train"]["text"][i * datanum: (i + 1) * datanum]
        data = list(filter(filter_nonenglish_text, data))
        word_freq += process(data, num_process, tokenizer)
    remainder = len(dataset["train"]["text"]) % datanum
    data = dataset["train"]["text"][-remainder:]
    data = list(filter(filter_nonenglish_text, data))
    word_freq += process(data, num_process, tokenizer)
    top_tokens = sorted(word_freq.items(), key=lambda item: (item[1], item[0]), reverse=True)
    with open(savepath+path.split("/")[-1]+'.txt', 'w') as file:
        for item in top_tokens:
            if tokenizer.convert_ids_to_tokens(item[0]) not in ['<unk>','<s>','</s>']:#filter <unk> <s> </s>,which may influence the alignment
                file.write(str(item[0]) + '\n')

if os.path.exists("/home/byzeng/NIPS_Code/sorted_tokens")==False:
    os.mkdir("/home/byzeng/NIPS_Code/sorted_tokens")

model_path="chainyo/alpaca-lora-7b"
output_path="/home/byzeng/NIPS_Code/sorted_tokens/"
sort_tokens_frequency(model_path,output_path)