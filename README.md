# -
智能养老利用情感识别技术和自然语言处理,开发出能够理解和回应老年人情感需求的陪伴机器人,为老年人提供情感慰藉和生活assistance。
from transformers import pipeline
import random

# Load emotion recognition model
emotion_pipeline = pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

# Load conversational model
chat_pipeline = pipeline('conversational', model='microsoft/DialoGPT-medium')

def get_emotion(text):
    """
    Recognize emotion from the text
    """
    results = emotion_pipeline(text)
    # Sorting to get the highest probability emotion
    sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)
    return sorted_results[0]['label']

def generate_response(text):
    """
    Generate a response based on the input text
    """
    chat_input = chat_pipeline(text)
    return str(chat_input)

def companion_robot():
    """
    Simulates interaction with the elderly companion robot
    """
    print("Hello! I'm here to keep you company. How are you feeling today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Robot: It was nice talking to you. Goodbye!")
            break
        
        emotion = get_emotion(user_input)
        print(f"Robot detects your emotion as {emotion}.")
        
        response = generate_response(user_input)
        print(f"Robot: {response}")

if __name__ == "__main__":
    companion_robot()
