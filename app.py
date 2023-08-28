from langchain.llms import Clarifai
import streamlit as st
from langchain import PromptTemplate, LLMChain

# Define the ProfitPilot class
def __init__(self, ai_name, ai_role, external_tools, company_name, company_values, conversation_type, conversation_purpose, company_business, salesperson_name, human_in_the_loop, prospect_name):
    self.ai_name = ai_name
    self.ai_role = ai_role
    self.external_tools = external_tools
    self.company_name = company_name
    self.company_values = company_values
    self.conversation_type = conversation_type
    self.conversation_purpose = conversation_purpose
    self.company_business = company_business
    self.salesperson_name = salesperson_name
    self.human_in_the_loop = human_in_the_loop
    self.prospect_name = prospect_name

# Variables setup
AI_NAME = "Athena"
EXTERNAL_TOOLS = None
COMPANY_NAME = "ABC Company"
COMPANY_VALUES = "Quality, Innovation, Customer Satisfaction"
CONVERSATION_TYPE = "Cold Email"  
CONVERSATION_PURPOSE = "discuss our new product"
COMPANY_BUSINESS = "APAC AI"
SALESPERSON_NAME = "John Doe"
HUMAN_IN_THE_LOOP = False
PROSPECT_NAME = "Jane Smith"


AI_ROLE = f"""
        You're the best cold emailer of APAC AI, you follow the principles of these books: SPIN Selling, To sell is Human, and FANATICAL Prospecting

        Never forget your name is {AI_NAME}. You work as a sales person.
        You work at company named {COMPANY_NAME}. {COMPANY_NAME}'s business is the following: {COMPANY_BUSINESS}.
        Company values are the following. {COMPANY_VALUES}
        You are contacting a potential prospect in order to {CONVERSATION_PURPOSE}
        Your means of contacting the prospect is {CONVERSATION_TYPE}

        If you're asked about where you got the user's contact information, say that you got it from public records.
        Keep your responses in short length to retain the user's attention. Never produce lists, just answers.
        Start the conversation by just a greeting and how is the prospect doing without pitching in your first turn.
        When the conversation is over, output 
        Always think about at which conversation stage you are at before answering:

        1: Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Always clarify in your greeting the reason why you are calling.
        2: Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.
        3: Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.
        4: Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.
        5: Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.
        6: Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.
        7: Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.
        8: End conversation: The prospect has to leave to call, the prospect is not interested, or next steps where already determined by the sales agent.

        Example 1:
        Conversation history:
        {SALESPERSON_NAME}: Hey, good morning! 
        User: Hello, who is this? <END_OF_TURN>
        {SALESPERSON_NAME}: This is {SALESPERSON_NAME} calling from {COMPANY_NAME}. How are you? 
        User: I am well, why are you calling?
        {SALESPERSON_NAME}: I am calling to talk about options for your home insurance. 
        User: I am not interested, thanks. 
        {SALESPERSON_NAME}: Alright, no worries, have a good day!
        End of example 1.

        You must respond according to the previous conversation history and the stage of the conversation you are at.
        Only generate one response at a time and act as {SALESPERSON_NAME} only! When you are done generating, end with to give the user a chance to respond.
"""# Create an instance of the ProfitPilot class

# Setup Clarifai with LangChain
CLARIFAI_PAT = '906eb260478642778e943dff45f66f3e'
USER_ID = 'meta'
APP_ID = 'Llama-2'
MODEL_ID = 'llama2-70b-chat'

template = AI_ROLE + """User input: {question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])

clarifai_llm = Clarifai(
    pat=CLARIFAI_PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID
)

llm_chain = LLMChain(prompt=prompt, llm=clarifai_llm)

# Streamlit chat interface
def clear_chat():
    st.session_state.messages = [{"role": AI_ROLE, "content": "Say something to get started!"}]

st.title("Profit Pilot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": AI_ROLE, "content": "Say something to get started!"}]

with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([4, 1])
    user_prompt = a.text_input(
        label="Your message:",
        placeholder="Type something...",
        label_visibility="collapsed",
    )
    b.form_submit_button("Send", use_container_width=True)

for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else AI_ROLE
    with st.chat_message(role):
        st.write(msg["content"])

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    response = llm_chain.run(user_prompt)  # If you want to use llm_chain

    msg = {"role": AI_ROLE, "content": response}
    st.session_state.messages.append(msg)

    with st.chat_message(AI_ROLE):
        st.write(msg["content"])

# Clear chat functionality
if len(st.session_state.messages) > 10:  # Assuming you want to clear chat after 10 messages
    st.button('Clear Chat', on_click=clear_chat)
