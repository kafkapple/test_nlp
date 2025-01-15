import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from models.dialogue_summarizer import (
    DialogueSummarizer, 
    DialogueSummarizerConfig,
    TokenizerConfig
)
from data.dataset import DataProcessor
from utils.metrics import compute_metrics
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

@hydra.main(version_base="1.2", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.general.seed)
    
    # Initialize model
    model_config = DialogueSummarizerConfig(
        model_name=cfg.model.name,
        tokenizer=TokenizerConfig(
            bos_token=cfg.model.tokenizer.bos_token,
            eos_token=cfg.model.tokenizer.eos_token,
            decoder_max_len=cfg.model.tokenizer.decoder_max_len,
            encoder_max_len=cfg.model.tokenizer.encoder_max_len,
            special_tokens=cfg.model.tokenizer.special_tokens
        )
    )
    model = DialogueSummarizer(model_config)
    
    # Prepare data
    processor = DataProcessor(model.tokenizer, model_config)
    train_dataset = processor.prepare_data(f"{cfg.general.data_path}/train.csv")
    val_dataset = processor.prepare_data(f"{cfg.general.data_path}/dev.csv")
    
    # 데이터셋 유효성 검사 추가
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty!")
        
    # 데이터셋 형식 확인
    sample = train_dataset[0]
    required_keys = ['input_ids', 'attention_mask', 'labels']
    missing_keys = [key for key in required_keys if key not in sample]
    if missing_keys:
        raise ValueError(f"Dataset is missing required keys: {missing_keys}")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.general.output_dir,
        num_train_epochs=cfg.train.training.num_epochs,
        learning_rate=cfg.train.training.learning_rate,
        per_device_train_batch_size=cfg.train.training.train_batch_size,
        per_device_eval_batch_size=cfg.train.training.eval_batch_size,
        warmup_ratio=cfg.train.training.warmup_ratio,
        weight_decay=cfg.train.training.weight_decay,
        fp16=cfg.train.training.fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: compute_metrics(model.tokenizer, p, cfg)
    )
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main() 