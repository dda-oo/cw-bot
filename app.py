import streamlit as st
import transformers

# Simple chatbot using GPT-2 for demo
def generate_response(prompt):
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

st.title("Simple GPT-2 Chatbot")
user_input = st.text_input("Ask me something:", "")

if user_input:
    response = generate_response(user_input)
    st.write(response)