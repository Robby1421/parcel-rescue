import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
from PIL import Image

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ParcelRescue: Your AI Solution for Parcel Management", page_icon="", layout="wide")

with st.sidebar:
    #st.image(' ')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container():
        l, m, r = st.columns((1, 3, 1))
        with l: st.empty()
        with m: st.empty()
        with r: st.empty()

    options = option_menu(
        "Dashboard",
        ["Home", "About Me", "Parcel Management", "Rescue Parcel"],
        icons=['book', 'globe', 'package', 'package'],
        menu_icon="book",
        default_index=0,
        styles={
            "icon": {"color": "#dec960", "font-size": "20px"},
            "nav-link": {"font-size": "17px", "text-align": "left", "margin": "5px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#262730"}
        })

if 'messagess' not in st.session_state:
    st.session_state.messagess = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options: Home
if options == "Home":
    st.title("Welcome to ParcelRescue, your AI-driven solution for efficient parcel management.")
    st.write("Designed to ensure seamless parcel tracking, recovery, and analysis, ParcelRescue leverages cutting-edge AI technology to handle your parcel logistics effortlessly.")
    st.write("From identifying delays to analyzing delivery trends, ParcelRescue empowers you with the tools to optimize your parcel operations and enhance customer satisfaction.")
    st.write("Experience the future of logistics management with ParcelRescue, where precision meets innovation.")

elif options == "About Me":
    st.title("About Me")
    st.write("# Robby Jean Pombo")
    st.write("## AI Engineer at Accenture Philippines")
    st.text("Connect with me via Linkedin : https://www.linkedin.com/in/robbyjeanpombo/")
    st.text("Github : https://github.com/robby1421/")
    st.write("\n")

# Options: Parcel Management
elif options == "📦 Parcel Management":
    dataframed = pd.read_excel("ParcelRescue.xlsx", engine="openpyxl")
    dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    documents = dataframed['combined'].tolist()
    embeddings = [get_embedding(doc, engine="text-embedding-3-small") for doc in documents]
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)

    System_Prompt = """
Role:
You are an intelligent assistant specialized in parcel management and logistics. Your role is to create, manage, and provide insights on parcel datasets to enhance delivery operations and customer satisfaction.

Instruct:
Your task is to assist in tracking parcels, identifying delays, and providing recommendations or visual insights (e.g., charts and summaries). When requested, locate specific parcels, identify delivery bottlenecks, and analyze status trends.

Context:
The company handles a wide range of parcels across multiple destinations with varying delivery timelines. Accurate parcel tracking and recovery help ensure timely deliveries, minimize delays, and maintain customer trust.

Constraints:
Ensure data is accurate, up-to-date, and formatted consistently.
Use clear, understandable language when explaining data or providing instructions.
Avoid assumptions about parcel details unless explicitly provided.

Examples:
Task: Identify delayed parcels.

Response: "Parcels such as ID 1003 (expected delivery: 2024-01-07) are delayed. The expected delivery date has passed without a delivery confirmation."

Task: Track a specific parcel.

Response: "Parcel ID 2021 is currently in transit and is expected to be delivered on 2024-01-10."

By following these guidelines, you will effectively support parcel management operations and enhance overall efficiency.
"""

    def initialize_conversation(prompt):
        if 'messagess' not in st.session_state:
            st.session_state.messagess = []
            st.session_state.messagess.append({"role": "system", "content": System_Prompt})
            chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages=st.session_state.messagess, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
            response = chat.choices[0].message.content
            st.session_state.messagess.append({"role": "assistant", "content": response})

    initialize_conversation(System_Prompt)

    for messages in st.session_state.messagess:
        if messages['role'] == 'system':
            continue
        else:
            with st.chat_message(messages["role"]):
                st.markdown(messages["content"])

    if user_message := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.markdown(user_message)
        query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
        query_embedding_np = np.array([query_embedding]).astype('float32')
        _, indices = index.search(query_embedding_np, 20)
        retrieved_docs = [documents[i] for i in indices[0]]
        context = ' '.join(retrieved_docs)
        structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"
        chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages=st.session_state.messagess + [{"role": "user", "content": structured_prompt}], temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
        st.session_state.messagess.append({"role": "user", "content": user_message})
        response = chat.choices[0].message.content
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messagess.append({"role": "assistant", "content": response})

# Options: Rescue Parcel
elif options == "Rescue Parcel":
    st.title("🚚 Rescue Parcel")
    st.write("This module helps in identifying and resolving issues related to delayed or misplaced parcels.")

    uploaded_file = '/mnt/data/ParcelRescue.xlsx'  # Replace with dynamic handling if required

    if os.path.exists(uploaded_file):
        parcel_data = pd.read_excel(uploaded_file)

        st.write("### Uploaded Parcel Data")
        st.dataframe(parcel_data.head())

        rescue_choice = st.selectbox("Choose an operation", ["Identify Delayed Parcels", "Track Parcel by ID", "Analyze Parcel Status"])

        if rescue_choice == "Identify Delayed Parcels":
            if 'Delivery Status' in parcel_data.columns and 'Expected Delivery Date' in parcel_data.columns:
                delayed_parcels = parcel_data[(parcel_data['Delivery Status'] != 'Delivered') & (parcel_data['Expected Delivery Date'] < pd.Timestamp.now())]
                st.write("### Delayed Parcels")
                st.dataframe(delayed_parcels)
            else:
                st.error("The dataset does not have the required columns.")

        elif rescue_choice == "Track Parcel by ID":
            parcel_id = st.text_input("Enter Parcel ID")
            if parcel_id:
                tracked_parcel = parcel_data[parcel_data['Parcel ID'] == parcel_id]
                if not tracked_parcel.empty:
                    st.write("### Parcel Details")
                    st.dataframe(tracked_parcel)
                else:
                    st.warning("No parcel found with the provided ID.")

        elif rescue_choice == "Analyze Parcel Status":
            if 'Delivery Status' in parcel_data.columns:
                status_counts = parcel_data['Delivery Status'].value_counts()
                st.write("### Parcel Status Distribution")
                st.bar_chart(status_counts)
            else:
                st.error("The dataset does not have a 'Delivery Status' column.")
    else:
        st.error("Parcel data file not found. Please upload the correct file.")
