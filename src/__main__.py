from .chunking import chunk_corpus

def main(): 
    l = chunk_corpus("data/raw/vllm-0.10.1")
    for chunk in l:
        print(chunk.text)
        print("//////")

main()