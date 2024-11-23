# import necessary packages

import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl.create_default_https_context =  ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "Good morning", "Good evening", "How are you?"],
        "responses": ["Hello! How can I assist you with your automobile inquiry?", "Hi there! What information do you need about automobiles?", "Hey! Looking for some automobile details?"]
    },
    {
        "tag": "vehicle_types",
        "patterns": ["What types of vehicles do you have?", "Tell me about your vehicles", "What kinds of automobiles are available?"],
        "responses": ["We have sedans, SUVs, hatchbacks, trucks, electric vehicles, and more. What are you interested in?"]
    },
    {
        "tag": "price_inquiry",
        "patterns": ["What is the price of [car name]?", "How much does [car model] cost?", "Tell me the cost of [vehicle name]"],
        "responses": ["The price of [car name] starts at $XX,XXX. Would you like to know about financing options?"]
    },
    {
        "tag": "fuel_type",
        "patterns": ["What fuel types are available?", "Do you have electric cars?", "Are there petrol or diesel options?"],
        "responses": ["We offer petrol, diesel, hybrid, and fully electric vehicles. Which one interests you?"]
    },
    {
        "tag": "test_drive",
        "patterns": ["Can I book a test drive?", "How do I schedule a test drive?", "I want to test drive a car"],
        "responses": ["Sure! You can schedule a test drive by providing your preferred date, time, and car model."]
    },
    {
        "tag": "features",
        "patterns": ["What are the features of [car name]?", "Tell me the specifications of [model name]", "What are the key features?"],
        "responses": ["The [car name] comes with features like [feature list]. Would you like detailed specifications?"]
    },
    {
        "tag": "availability",
        "patterns": ["Is [car name] available?", "Do you have [model name] in stock?", "Can I buy [car name] now?"],
        "responses": ["Yes, the [car name] is available in our showroom. Would you like to visit or book it online?"]
    },
    {
        "tag": "finance_options",
        "patterns": ["Do you have financing options?", "Can I get a loan for the car?", "What are the EMI plans available?"],
        "responses": ["Yes, we offer flexible financing options and EMI plans. Would you like to discuss these in detail?"]
    },
    {
        "tag": "service_centers",
        "patterns": ["Where are your service centers?", "Do you have a service center near me?", "What about car servicing?"],
        "responses": ["Our service centers are located across various cities. Please provide your location for more details."]
    },
    {
        "tag": "warranty",
        "patterns": ["What is the warranty on [car name]?", "Do you provide a warranty?", "How long is the warranty period?"],
        "responses": ["The [car name] comes with a [X years] warranty. Would you like to know what it covers?"]
    },
    {
        "tag": "exchange",
        "patterns": ["Do you have an exchange offer?", "Can I trade in my old car?", "What about exchanging my vehicle?"],
        "responses": ["Yes, we offer exchange programs. Please share your current car details for an evaluation."]
    },
    {
        "tag": "location",
        "patterns": ["Where is your showroom?", "What is the location of your dealership?", "How can I reach your showroom?"],
        "responses": ["Our showroom is located at [address]. Would you like directions or assistance with visiting us?"]
    },
    {
        "tag": "farewell",
        "patterns": ["Bye", "Goodbye", "See you later", "Thanks, bye"],
        "responses": ["Goodbye! Feel free to reach out for more automobile inquiries!", "Take care! Let us know if you need further assistance."]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks a lot", "I appreciate it", "Much appreciated"],
        "responses": ["You're welcome! Happy to help.", "No problem! Let us know if you have more questions.", "Glad to assist!"]
    }
]


# Create vectoriser and classifier

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

#preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns) #value
y = tags #label
clf.fit(x,y)


# Create Chatbot
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]

    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response


# Chatbot using Streamlit

counter = 0

def main():
    global counter
    st.title("AutoMate")
    st.write("Welcome to AutoMate")

    counter+=1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response=chatbot(user_input)
        st.text_area("AutoMate:", value=response, height=100,max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thanks")
            st.stop()


if __name__ == '__main__':
    main()


