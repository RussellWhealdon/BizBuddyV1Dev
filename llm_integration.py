from openai import OpenAI
import streamlit as st

# Initialize the OpenAI client
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Business context for session memory
business_context = """
Answer these questions based on this context: The data is from a one-person dietitian business that began about a year ago. The dietitian has some technical 
skills and seeks to use GA4 data to grow her website’s performance and make clear, actionable business decisions. Keep insights simple, specific, and free from jargon. 
Keep a few key things in mind, she is in lynnwood Washing just outside Seattle. She is hoping to work specifcally with Adults with Eating disorders. A conversion event 
for her is someone going to the contact page and filling out a contact form (a lead). Keep in mind this data is from this year summarized for that whole time period.
"""

def initialize_llm_context():
    if "session_summary" not in st.session_state:
        st.session_state["session_summary"] = business_context

def query_gpt(prompt, data_summary=""):
    try:
        session_summary = st.session_state.get("session_summary", "")
        full_prompt = f"{session_summary}\n\nData Summary:\n{data_summary}\n\nUser Question: {prompt}"

        # Send the prompt to GPT-4 through the OpenAI client instance
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst with a focus on digital growth and conversion optimization."},
                {"role": "user", "content": full_prompt}
            ]
        )
        
        # Access the response using dot notation
        answer = response.choices[0].message.content
        st.session_state["session_summary"] += f"\nUser: {prompt}\nModel: {answer}\n"
        
        return answer

    except Exception as e:
        return f"Error: {e}"


def query_gpt_keywordbuilder(prompt, data_summary=""):
    try:
        full_prompt = f"\n\nData Summary:\n{data_summary}\n\nUser Question: {prompt}"
    
        # Send the prompt to GPT-4 through the OpenAI client instance
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst with a focus on digital growth and conversion optimization."},
                {"role": "user", "content": full_prompt}
            ]
        )
        
        # Access the response using dot notation
        answer = response.choices[0].message.content
        
        return answer

    except Exception as e:
        return f"Error: {e}"
