import torch

def get_answer(question, context, tokenizer, model):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer_tokens = all_tokens[start_index:end_index + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer
