from rouge import Rouge

def compute_metrics(tokenizer, pred, config):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids
    
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    
    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)
    
    # Remove special tokens
    for token in config.inference.remove_tokens:
        decoded_preds = [sent.replace(token, " ") for sent in decoded_preds]
        labels = [sent.replace(token, " ") for sent in labels]
    
    results = rouge.get_scores(decoded_preds, labels, avg=True)
    return {key: value["f"] for key, value in results.items()} 