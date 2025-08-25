import os

# os.environ['HF_HUB_URL'] = 'https://hf-mirror.com'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.environ["XDG_CACHE_HOME"] = "/mnt/cache/code/.cache"

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset, concatenate_datasets

def apps():
    data = load_dataset('codeparrot/apps')

    data['train'].to_json('source/apps_train.jsonl')
    data['test'].to_json('source/apps_test.jsonl')
    
def livecodebench():
    data = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="v4_v5", trust_remote_code=True)
    data.to_json('test/livecodebench.jsonl')

def bigcodebench():
    data = load_dataset('bigcode/bigcodebench', split='v0.1.0_hf')
    data.to_json('test/bigcodebench.jsonl')

if __name__ == '__main__':
    apps()