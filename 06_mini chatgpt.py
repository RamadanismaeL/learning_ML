from transformers import AutoModelForCausalLM, AutoTokenizer

#1. Escolher modelo pequeno (menos recursos)
model_name = "distilgpt2" # versão leve do GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#2. Função para conversar
def mini_chat(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_k=50, #diversidade
        top_p=0.95
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

#3. Teste a conversa
while True:
    user_input = input("Você: ")
    if user_input.lower() in ["sair", "exit", "quit"]:
        break
    resposta = mini_chat(user_input)
    print("MiniChatGPT: ",resposta)