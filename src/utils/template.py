
qwen = dict(
    question="<|im_start|>system\nYou are an intelligent programming assistant to produce Python algorithmic solutions.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n",
    pre="<|fim_prefix|>", 
    suf="<|fim_suffix|>", 
    mid="<|fim_middle|>", 
    eot="<|endoftext|>",
    stop=['<|endoftext|>', '<|im_end|>']
)