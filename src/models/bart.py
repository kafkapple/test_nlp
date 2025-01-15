from dataclasses import dataclass
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch

@dataclass
class TokenizerConfig:
    bos_token: str
    eos_token: str
    decoder_max_len: int
    encoder_max_len: int
    special_tokens: dict

@dataclass
class DialogueSummarizerConfig:
    model_name: str
    tokenizer: TokenizerConfig
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class BartSummarizer:
    def __init__(self, config: DialogueSummarizerConfig):
        self.config = config
        self.device = config.device
        self._init_model()
        
    def _init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # special tokens를 딕셔너리로 안전하게 변환
        special_tokens = {
            'additional_special_tokens': [
                str(token) if not isinstance(token, str) else token 
                for token in self.config.tokenizer.special_tokens.additional_special_tokens
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        self.model = BartForConditionalGeneration.from_pretrained(self.config.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        
    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids=input_ids.to(self.device), **kwargs) 