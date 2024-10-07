import google.generativeai as genai

def initialize_model(api_key, model_name="gemini-pro"):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def get_chat_response(model, prompt):
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text
