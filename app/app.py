import streamlit as st

from server import stream_invoke


st.set_page_config(page_title="Chat with AI", page_icon=":speech_balloon:")
st.title("Live Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner():
        response = st.write_stream(stream_invoke(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
