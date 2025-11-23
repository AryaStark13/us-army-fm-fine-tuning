import streamlit as st
import torch
# from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

from dotenv import load_dotenv
load_dotenv()

device = torch.device("cuda")

# Page config
st.set_page_config(
    page_title="Army FM Chatbot",
    page_icon="🎖️",
    layout="centered"
)

@st.cache_resource
def load_model(model_name, max_seq_length=2048):
    """Load model and tokenizer - cached to avoid reloading"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, device=device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda"
    )
    # Works - Do not change
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device=device)
    # model = AutoModelForCausalLM.from_pretrained(
    #     "ShethArihant/Llama-3.1-8B-us-army-fm-instruct",
    #     dtype=torch.bfloat16,
    #     device_map="cuda"
    # )

    return model, tokenizer

def format_chat_messages(messages, tokenizer):
    """Format messages using the model's chat template"""
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    return inputs

def generate_response(model, tokenizer, messages, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """Generate streaming response from the model"""
    # Tokenize messages
    inputs = format_chat_messages(messages, tokenizer)
    
    # Setup streamer
    text_streamer = TextIteratorStreamer(tokenizer)
    
    # Generation kwargs
    generation_kwargs = dict(
        inputs,
        streamer = text_streamer,
        max_new_tokens = max_new_tokens,
        use_cache = True,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,  # Handle padding
        eos_token_id=tokenizer.eos_token_id,  # Explicit EOS
    )
    
    # Start generation in separate thread
    thread = Thread(target = model.generate, kwargs = generation_kwargs)
    thread.start()
    
    # Yield tokens as they're generated
    generation_length = 0
    for idx, new_text in enumerate(text_streamer):
        if idx != 0:
            generation_length += 1
            yield new_text
    
    print("Total number of tokens generated:", generation_length)
    print("Set Max New Tokens:", max_new_tokens)
    thread.join()

# Sidebar for configuration
with st.sidebar:
    st.title("🎖️ Army FM Chatbot")
    st.markdown("---")
    
    # Model selection
    model_name = st.text_input(
        "Model Name",
        value="ShethArihant/Llama-3.2-3B-army-text-instruct",
        help="HuggingFace model name"
    )
    
    # Generation parameters
    st.markdown("### Generation Settings")
    max_new_tokens = st.slider("Max New Tokens", 64, 4096, 2048, 64)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)
    
    st.markdown("---")
    
    # Export chat button
    if len(st.session_state.get("messages", [])) > 0:
        # Convert messages to JSON
        chat_json = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="📥 Export Chat",
            data=chat_json,
            file_name="army_fm_chat_export.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.button("📥 Export Chat", disabled=True, use_container_width=True, 
                 help="No messages to export yet")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This chatbot uses a Llama model fine-tuned on US Army Field Manuals.
    
    Ask questions about:
    - Military doctrine
    - Field operations
    - Army procedures
    - Tactical concepts
    """)

# Main chat interface
st.title("💬 Chat with Army FM Model")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load model
try:
    with st.spinner("Loading model..."):
        model, tokenizer = load_model(model_name)
    st.success("Model loaded!", icon="✅")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about Army Field Manuals..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        try:
            for token in generate_response(
                model, 
                tokenizer, 
                st.session_state.messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            ):
                full_response += token
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            message_placeholder.error(f"Error generating response: {e}")
            full_response = f"Error: {e}"
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Built with Streamlit • Powered by Unsloth • Model: Llama 3.2 3B
    </div>
    """,
    unsafe_allow_html=True
)