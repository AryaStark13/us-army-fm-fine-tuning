import streamlit as st
import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread
import os

# Page config
st.set_page_config(
    page_title="Army FM Chatbot",
    page_icon="🎖️",
    layout="centered"
)

@st.cache_resource
def load_model(model_name, max_seq_length=2048):
    """Load model and tokenizer - cached to avoid reloading"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        token=os.environ.get('HF_TOKEN'),
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def format_chat_messages(messages, tokenizer):
    """Format messages using the model's chat template"""
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted

def generate_response(model, tokenizer, messages, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """Generate streaming response from the model"""
    # Format messages
    prompt = format_chat_messages(messages, tokenizer)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Setup streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generation kwargs
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        use_cache=True,
    )
    
    # Start generation in separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield tokens as they're generated
    for new_text in streamer:
        yield new_text
    
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
    max_new_tokens = st.slider("Max New Tokens", 64, 2048, 512, 64)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)
    
    st.markdown("---")
    
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