import streamlit as st
import asyncio
import json
from orchestrator import example_usage_2

# Page configuration
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Guardian AI")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0  # counter for unique widget IDs

# Sidebar container for uploader only
uploader_container = st.sidebar.empty()

def render_uploader():
    with uploader_container.container():
        uploaded = st.file_uploader(
            "Upload an Image, Audio, or Video",
            type=["jpg", "jpeg", "png", "mp3", "wav", "mp4", "mov", "avi"],
            key=f"uploader_{st.session_state.uploader_key}",  # unique key
        )
        if uploaded:
            st.session_state.uploaded_file = uploaded
        
        # Show preview only if we have a file in session state
        if st.session_state.uploaded_file:
            file_type = st.session_state.uploaded_file.type
            # Show preview until message is sent
            if "image" in file_type:
                st.image(st.session_state.uploaded_file, caption="Selected Image")
            elif "audio" in file_type:
                st.audio(st.session_state.uploaded_file)
            elif "video" in file_type:
                st.video(st.session_state.uploaded_file)
        return uploaded

uploaded_file = render_uploader()

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.markdown(msg["content"])
        elif msg["type"] == "media":
            file = msg["content"]
            if "image" in file.type:
                st.image(file, caption="Uploaded Image")
            elif "audio" in file.type:
                st.audio(file)
            elif "video" in file.type:
                st.video(file)

# Text input box
if prompt := st.chat_input("Type your message..."):
    # Add user's message to conversation history
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

        
        # return results
        response = asyncio.run(example_usage_2(r"C:\Users\PRYth\OneDrive\Desktop\Agentic\SimplifyAgentic\test_input\example3.mp4", prompt))
        # response = json.dumps(response, indent=2)

        response = f"""
        ### 📝 Scam Analysis Report

        **Summary:**  
        {response['summary']}

        **Scores:**  
        - 🤖 AI Score: {response['ai_score_0_100']}/100  
        - 🕵️ Fake Score: {response['fake_score_0_100']}/100  
        - 📊 Confidence: {response['confidence_0_100']}/100  

        **Key Evidence:**  
        """ + "\n".join([f"- {e}" for e in response['key_evidence']]) + """

        **Recommendations:**  
        """ + "\n".join([f"- {r}" for r in response['recommendations']])
    # Handle file upload if one exists in session state
    if st.session_state.uploaded_file:

        st.session_state.uploader_key += 1  # Increment the key to force a reset of the file uploader
        st.session_state.uploaded_file = None  # Clear the uploaded file from session state

        # response = "File uploaded with your message!" ### update here
    # else:
        # response = f"Echo: {prompt}"

    # Add assistant's response to conversation history and display it
    st.session_state.messages.append({"role": "assistant", "type": "text", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    # Rerun the app to re-render the sidebar after updating the session state
    st.rerun()