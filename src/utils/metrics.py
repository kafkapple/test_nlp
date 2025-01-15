from rouge import Rouge

def compute_metrics(tokenizer, pred, config):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    # train/default.yaml의 remove_tokens 설정 사용
    for token in config.train.inference.remove_tokens:
        pred_str = [pred.replace(token, '') for pred in pred_str]
        label_str = [label.replace(token, '') for label in label_str]
    
    rouge = Rouge()
    scores = rouge.get_scores(pred_str, label_str, avg=True)
    
    return {
        'rouge1_f1': scores['rouge-1']['f'],
        'rouge2_f1': scores['rouge-2']['f'],
        'rougeL_f1': scores['rouge-l']['f']
    } 