import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("🔄 Memuat model...")

base_model_id = "unsloth/gemma-2-2b-it"
adapter_id = "JSpiritGG/jogja-kuliner-chatbot"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

print("✅ Model siap!")

template = "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"

def chat(message, history):
    prompt = template.format(question=message)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "model\n" in response:
        response = response.split("model\n")[-1].strip()
    return response

demo = gr.ChatInterface(
    fn=chat,
    title="🍛 Asisten Kuliner & Sejarah Jogja",
    description="Chatbot Fine-tuned Gemma 2B dengan LoRA - UAS Deep Learning",
    examples=["Apa itu gudeg?", "Rekomendasi kuliner malam di Jogja?", "Siapa pendiri Muhammadiyah?"],
    theme="soft"
)

demo.launch()
