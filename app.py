import streamlit as st
import asyncio
import json
import os
import tempfile
from orchestrator import example_usage_2
from datetime import datetime
import re

def extract_and_print_fields(json_string):
        # Define the fields we want to extract
        fields = ["text", "audio", "visual", "technical"]
        res = {}
        for field in fields:
            # Regex pattern to match "field": "content" (handles escaped quotes and multiline)
            pattern = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)\"'
            match = re.search(pattern, json_string, re.DOTALL)
            
            if match:
                content = match.group(1)
                # Unescape any escaped quotes
                content = content.replace('\\"', '"')
                res[field] = content
                print(f'"{field}": {content}')
            else:
                print(f'"{field}": Not found')
        return res

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
        tmp_file_path = None
        if uploaded:
            st.session_state.uploaded_file = uploaded
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp_file:
                tmp_file.write(uploaded.read())
                tmp_file_path = tmp_file.name
        
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
        return uploaded,tmp_file_path

uploaded_file,tmp_file_path = render_uploader()

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
    time_before = datetime.now()

    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
        # return results
        with st.spinner("🤖 Processing your request..."):
            # response = asyncio.run(example_usage_2(prompt, example_video_path=r"C:\Users\ASUS\Downloads\agentic workshop 2\SimplifyAgentic\test_input\example3.mp4"))[0]
            response = asyncio.run(example_usage_2(prompt, example_video_path=tmp_file_path))[0]
            st.success("Done!")
        # response = json.dumps(response, indent=2)
        
        # print(type(response))
        # print(response)
        if response:
            for key,val in response.items():
                print(f"{key}: {val}\n")
            final_response = f"""
📝 Final Report

**Summary:**  
{response['summary']}

**Scores:**  
- 🤖 AI Score: {response['ai_score_0_100']}/100  
- 🕵️ Fake Score: {response['fake_score_0_100']}/100  
- 📊 Confidence: {response['confidence_0_100']}/100  
            
"""
            keys = response.keys()
            # print(keys)
            # print(type(response["key_evidence"]))
            # print(type(response["component_analysis"]))
            # print("types")
            if "key_evidence" in keys:
                
                # print(response["key_evidence"])
                final_response += "**Key Evidence:** \n"
                if isinstance(response["key_evidence"], list):
                    for i, item in enumerate(response["key_evidence"]):
                        final_response += f""" {item} \n"""

                elif isinstance(response["key_evidence"], str):
                    final_response += response["key_evidence"]
                    final_response += "\n"

            if "component_analysis" in keys:
                
                # print(response["component_analysis"])
                final_response+= "\n**Component Analysis:** \n"
                print(type(response["component_analysis"]))
                if isinstance(response["component_analysis"], list):
                    for i, item in enumerate(response["component_analysis"]):
                        final_response += f""" {item} \n"""

                elif isinstance(response["component_analysis"], str):
                    try:
                        # extract_and_print_fields(json_data)
                        extracted_fields =  extract_and_print_fields(response["component_analysis"])
                        for field in extracted_fields:

                            final_response += f"\n {response["component_analysis"]}"

                        # formatted_string = "{" + response["component_analysis"].strip().replace(' "', '", "') + "}"
                        # data = json.loads(formatted_string)

                        # for key, value in data.items():
                        #     # Capitalize the key and replace underscore for better readability
                        #     formatted_key = key.replace('_', ' ').title()
                        #     final_response += f"{formatted_key}: {value}\n"

                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        final_response += response["component_analysis"]
                        final_response += "\n"
                elif isinstance(response["component_analysis"], dict):
                    for key,value in response["component_analysis"].items():
                        final_response+= f"{key}: {value}\n"
            
            
    # Handle file upload if one exists in session state
    if st.session_state.uploaded_file:

        st.session_state.uploader_key += 1  # Increment the key to force a reset of the file uploader
        st.session_state.uploaded_file = None  # Clear the uploaded file from session state


    # Add assistant's response to conversation history and display it
    st.session_state.messages.append({"role": "assistant", "type": "text", "content": final_response})
    with st.chat_message("assistant"):
        print(f"Time elapsed: {(datetime.now()-time_before).seconds}")
        # final_response += f"\nTime elapsed: {(datetime.now()-time_before).seconds}"
        st.markdown(final_response)

    # Rerun the app to re-render the sidebar after updating the session state
    st.rerun()