from langchain.llms import Clarifai
import streamlit as st
from langchain import PromptTemplate, LLMChain
# Setup Clarifai with LangChain
CLARIFAI_PAT = '906eb260478642778e943dff45f66f3e'
USER_ID = 'meta'
APP_ID = 'Llama-2'
# Change these to whatever model and text URL you want to use
MODEL_ID = 'llama2-7b-chat'

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

clarifai_llm = Clarifai(
    pat=CLARIFAI_PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID
)

llm_chain = LLMChain(prompt=prompt, llm=clarifai_llm)

# Streamlit chat interface
def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Say something to get started!"}]

st.title("Profit Pilot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Say something to get started!"}]

with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([4, 1])
    user_prompt = a.text_input(
        label="Your message:",
        placeholder="Type something...",
        label_visibility="collapsed",
    )
    b.form_submit_button("Send", use_container_width=True)

for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(msg["content"])

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    response = llm_chain.run(user_prompt)

    msg = {"role": "assistant", "content": response}
    st.session_state.messages.append(msg)

    with st.chat_message("assistant"):
        st.write(msg["content"])

if len(st.session_state.messages) > 1:
    st.button('Clear Chat', on_click=clear_chat)
