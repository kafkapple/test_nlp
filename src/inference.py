import hydra
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from data.dataset import DataProcessor
from models.dialogue_summarizer import DialogueSummarizerConfig, TokenizerConfig, DialogueSummarizer
from omegaconf import DictConfig
import os
import pandas as pd

class DialogueInference:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.model_config = self._create_model_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def _load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        
        # special tokens를 문자열로 변환하여 처리
        special_tokens = [str(token) for token in self.cfg.model.tokenizer.special_tokens.additional_special_tokens]
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        
        # 특수 토큰 추가
        tokenizer.add_special_tokens(special_tokens_dict)
        
        model = BartForConditionalGeneration.from_pretrained(self.cfg.inference.ckt_path)
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer
    
    def _create_model_config(self):
        return DialogueSummarizerConfig(
            model_name=self.cfg.model.name,
            tokenizer=TokenizerConfig(
                bos_token=self.cfg.model.tokenizer.bos_token,
                eos_token=self.cfg.model.tokenizer.eos_token,
                decoder_max_len=self.cfg.model.tokenizer.decoder_max_len,
                encoder_max_len=self.cfg.model.tokenizer.encoder_max_len,
                special_tokens=self.cfg.model.tokenizer.special_tokens
            )
        )
    
    def generate_summaries(self, test_dataset):
        self.model.eval()
        dataloader = DataLoader(test_dataset, batch_size=self.cfg.inference.batch_size)
        
        summaries = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    no_repeat_ngram_size=self.cfg.inference.no_repeat_ngram_size,
                    early_stopping=self.cfg.inference.early_stopping,
                    max_length=self.cfg.inference.generate_max_length,
                    num_beams=self.cfg.inference.num_beams,
                )
                
                decoded_summaries = [self.tokenizer.decode(g, skip_special_tokens=True) 
                                   for g in generated_ids]
                summaries.extend(decoded_summaries)
                
        return summaries
    
    def inference(self, test_file_path):
        # 데이터 준비
        processor = DataProcessor(self.tokenizer, self.model_config)
        test_dataset = processor.prepare_data(test_file_path, is_train=False)
        
        # 요약문 생성
        summaries = self.generate_summaries(test_dataset)
        
        # 결과 저장
        test_df = pd.read_csv(test_file_path)
        results = pd.DataFrame({
            'fname': test_df['fname'],
            'summary': summaries
        })
        
        os.makedirs(self.cfg.general.output_dir, exist_ok=True)
        output_path = f"{self.cfg.general.output_dir}/predictions.csv"
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        return results

@hydra.main(version_base="1.2", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    inferencer = DialogueInference(cfg)
    test_file_path = f"{cfg.general.data_path}/test.csv"
    results = inferencer.inference(test_file_path)

if __name__ == "__main__":
    main() 