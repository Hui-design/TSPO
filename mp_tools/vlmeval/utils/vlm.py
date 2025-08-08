import os
import pandas as pd

def build_option_str(option_dict):
    s = 'There are several options: \n'
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s

def gpt_key_set():
    openai_key = os.environ.get('OPENAI_API_KEY', None)
    return isinstance(openai_key, str) and openai_key.startswith('sk-')
