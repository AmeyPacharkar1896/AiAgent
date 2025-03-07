from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
lm_model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt, max_length=100):
    """
    Generate a text response given a prompt.
    If the prompt is empty, return a default message.
    """
    # Check if the prompt is empty or whitespace only
    if not prompt.strip():
        return "I didn't catch that. Could you please repeat?"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Check if tokenization resulted in empty input_ids
    if inputs.input_ids.size(1) == 0:
        return "I didn't catch that. Could you please repeat?"
    
    outputs = lm_model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id  # ensure proper padding for generation
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
