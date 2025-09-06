#!/usr/bin/env python3
"""
Nepali Name Generator - Streamlit Web Interface

A beautiful and interactive web interface for generating Nepali names using 
trained neural network models.
"""

import streamlit as st
import sys
import os
import torch
from pathlib import Path
import time
import random

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import project modules
from config import Config, DataConfig, TrainingConfig, SamplingConfig, SystemConfig
from models import Transformer
from data import create_datasets
from utils import generate

# Page configuration
st.set_page_config(
    page_title="🏔️ Nepali Name Generator",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# initialize help flag in session state
if 'show_help' not in st.session_state:
    st.session_state.show_help = False

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Base typography and theme-aware defaults */
    body, .stApp {
        font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        color: #f5f7fa; /* light text as default for dark Streamlit theme */
    }

    /* Header: simplified flexbox for perfect centering */
    .main-header {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin: 0 auto 0.6rem auto;
        letter-spacing: -0.3px;
        width: fit-content;
    }

    .subtitle {
        text-align: center;
        color: rgba(245,247,250,0.9);
        font-size: 1.06rem;
        margin-bottom: 1.2rem;
    }

    /* Hero wrapper to constrain header and controls so they visually center together */
    .hero {
        max-width: 680px;
        width: 100%;
        margin: 0 auto;
        text-align: center;
        box-sizing: border-box;
        padding: 0 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* Constrain header and subtitle to the hero width so the button aligns visually */
    .main-header, .subtitle {
        width: 100%;
        text-align: center;
        margin-left: auto;
        margin-right: auto;
    }

    /* Force button container and button to center properly */
    .hero .stButton,
    .hero div[data-testid="stVerticalBlock"] > div > div,
    .hero div[data-testid="baseButton-secondary"] {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
    }

    /* CTA block to keep button and helper text perfectly centered together */
    .cta {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        width: fit-content;
        margin: 0 auto 1rem auto;
        text-align: center;
    }
    .cta .helper {
        color: rgba(255,255,255,0.72);
        font-size: 0.95rem;
        line-height: 1.35;
        max-width: 640px;
    }

    /* Name tiles - premium glass gradients */
    .name-card {
        position: relative;
        background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.08);
        color: #f7fbff;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        margin: 0.5rem 0.35rem;
        text-align: center;
        font-size: 1.08rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        text-shadow: 0 1px 2px rgba(0,0,0,0.35);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        transition: transform 0.22s ease, box-shadow 0.22s ease, filter 0.22s ease;
        min-height: 52px;
    }

    .male-card { background: linear-gradient(135deg, #00D2FF 0%, #3A47D5 100%); }
    .female-card { background: linear-gradient(135deg, #FF6CAB 0%, #7366FF 100%); }
    .name-card:hover { transform: translateY(-6px) scale(1.015); box-shadow: 0 18px 50px rgba(0,0,0,0.45); filter: saturate(1.05); }

    /* Light blocks (stats, sidebar, generation info) should have dark text */
    .stats-container, .generation-info, .sidebar-content {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #0f1723; /* dark text for light panels */
        border: 1px solid #e6e9ee;
    }

    .generation-info { border-left: 4px solid #007bff; background: #f1f9ff; }

    /* Buttons - simple, robust centering */
    .stButton > button {
        background: linear-gradient(90deg, #4ECDC4 0%, #FF6B6B 100%) !important;
        color: #062827 !important;
        border: none !important;
        padding: 0.85rem 2rem !important;
        font-size: 1.08rem !important;
        border-radius: 16px !important;
        font-weight: 800 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 14px 40px rgba(78,204,196,0.12) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        transition: transform 0.14s ease, box-shadow 0.14s ease !important;
        margin: 0 auto !important;   /* center inside its own block */
        display: block !important;
        width: fit-content !important;
    }

    .stButton > button:hover { 
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 20px 55px rgba(78,204,196,0.16) !important;
    }

    /* Generate area styling */
    .generate-card {
        display: flex;
        gap: 1rem;
        align-items: center;
    }

    .info-panel {
        background: linear-gradient(180deg, rgba(17,24,39,0.95), rgba(15,23,42,0.85));
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: rgba(255,255,255,0.95);
        box-shadow: 0 8px 30px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.04);
        font-weight: 500;
        line-height: 1.35;
    }

    .info-panel .muted { color: rgba(255,255,255,0.74); font-weight: 400; font-size: 0.95rem; }

    /* Remove duplicate/conflicting button styles */

    /* Center info panel content and match height with button area */
    .generate-card { align-items: center; display: flex; gap: 1rem; justify-content: center; }
    .info-panel { display: flex; align-items: center; justify-content: center; }

    /* Ensure markdown and text widgets inherit correct color depending on container */
    .stMarkdownContainer, div[data-testid="stVerticalBlock"], [data-testid*="markdown"] {
        color: inherit !important;
    }

    /* Headings: inherit color and subtle shadow for legibility */
    h1, h2, h3, h4, h5, h6 {
        color: inherit;
        text-shadow: 0 1px 2px rgba(0,0,0,0.45);
        margin: 0.25rem 0;
    }

    /* Animations and micro-interactions (kept) */
    .skeleton { background: linear-gradient(90deg, #e6e6e6 25%, #dddddd 50%, #e6e6e6 75%); background-size: 200% 100%; animation: loading 1.5s infinite; border-radius: 8px; }
    @keyframes loading { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

    /* Name cards: center text and ensure consistent vertical alignment */
    .name-card { transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1); cursor: pointer; display: flex; align-items: center; justify-content: center; min-height: 52px; }
    .name-card:hover { transform: translateY(-6px) scale(1.02); box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45); }

    .typing-animation { border-right: 2px solid #FF6B6B; animation: blink 1s infinite; white-space: nowrap; overflow: hidden; }
    @keyframes blink { 0%,50% { border-color: #FF6B6B; } 51%,100% { border-color: transparent; } }

    .progress-bar { width: 100%; height: 8px; background-color: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden; margin: 1rem 0; }
    .progress-fill { height: 100%; background: linear-gradient(90deg, #FF6B6B, #4ECDC4); border-radius: 4px; animation: progress 2s ease-in-out; }
    @keyframes progress { 0% { width: 0%; } 100% { width: 100%; } }

    .name-reveal { animation: fadeInUp 0.6s ease-out forwards; opacity: 0; transform: translateY(20px); }
    .name-static { opacity: 1; transform: translateY(0); }
    .name-reveal:nth-child(1) { animation-delay: 0.08s; }
    .name-reveal:nth-child(2) { animation-delay: 0.16s; }
    .name-reveal:nth-child(3) { animation-delay: 0.24s; }
    .name-reveal:nth-child(4) { animation-delay: 0.32s; }
    .name-reveal:nth-child(5) { animation-delay: 0.40s; }
    @keyframes fadeInUp { to { opacity: 1; transform: translateY(0); } }

    .pulse-button { animation: pulse 2s infinite; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.6); } 70% { box-shadow: 0 0 0 10px rgba(255, 107, 107, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); } }

    /* small utility adjustments */
    .names-container { display: block; max-width: 1200px; margin: 0 auto; padding: 0 1rem; }
    .stats-container strong, .generation-info strong { font-weight: 700; }

    /* Ensure stats/info panels take full width of the sidebar column and align text nicely */
    .stats-container, .generation-info { width: 100%; box-sizing: border-box; }

    /* Reduce default top padding added by Streamlit so hero sits closer to the top */
    div[data-testid="stAppViewContainer"],
    div[data-testid="stAppViewContainer"] > div,
    div[data-testid="stAppViewContainer"] > div > div,
    div[data-testid="root"] > div > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Also reduce .block-container top padding */
    .main .block-container { padding-top: 0.4rem !important; }

    /* Top hero slight nudge */
    .hero.hero-top { margin-top: 0.25rem; }

    /* Section divider under hero */
    .section-divider { height: 1px; width: 92%; max-width: 900px; margin: 1rem auto 1.6rem; background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.14), rgba(255,255,255,0)); border-radius: 999px; }

    /* Nudge first markdown container up if Streamlit still applies spacing */
    div[data-testid="stMarkdownContainer"]:first-of-type {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* keep JS for typing/reveal intact (inlined) */
    </style>
""", unsafe_allow_html=True)

# Note: JavaScript for typing/reveal removed to avoid script tags rendering as text in Streamlit.
# Animations are handled by CSS-only rules in the stylesheet above.

@st.cache_resource
def load_model_and_data(gender_preference):
    """Load the trained model and prepare data with caching."""
    try:
        # Get data files based on gender preference
        if gender_preference.lower() == "male":
            data_files = ["data/male.txt"]
            model_dir = "models/male_finetuned"
        elif gender_preference.lower() == "female":
            data_files = ["data/female.txt"]
            model_dir = "models/female_finetuned"
        else:  # "both"
            data_files = ["data/male.txt", "data/female.txt"]
            model_dir = "models/best_results"

        # Load training data for vocabulary and block_size (use same data as training)
        temp_config = Config(data=DataConfig(input_files=data_files))
        train_dataset, _ = create_datasets(data_files, temp_config.data, use_lowercase=True)

        # Get model parameters
        vocab_size = train_dataset.get_vocab_size()
        block_size = train_dataset.get_output_length()

        # Initialize model
        model_config = Config().model
        model_config.vocab_size = vocab_size
        model_config.block_size = block_size

        model = Transformer(model_config)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # Load trained weights
        model_path = os.path.join(model_dir, "model.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model_final.pt")
            if not os.path.exists(model_path):
                return None, None, None, f"Model not found at {model_dir}"

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Create dataset for generation
        # Now create dataset for generation using the gender preference
        gen_dataset, _ = create_datasets(data_files, Config(data=DataConfig(input_files=data_files)).data, use_lowercase=True)

        return model, gen_dataset, {
            'vocab_size': vocab_size,
            'block_size': block_size,
            'model_path': model_path,
            'device': device,
            'parameters': sum(p.numel() for p in model.parameters())
        }, None

    except Exception as e:
        return None, None, None, str(e)

def generate_names(model, dataset, config, num_names, temperature, top_k, max_length, start_with=""):
    """Generate names using the model."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create initial context
        if start_with:
            # Encode the starting letter and use it as initial context
            start_token = dataset.encode(start_with.lower())[0]  # Get the token for the starting letter
            X_init = torch.full((num_names, 1), start_token, dtype=torch.long).to(device)
        else:
            X_init = torch.zeros(num_names, 1, dtype=torch.long).to(device)
        
        # Generate names
        top_k_val = top_k if top_k > 0 else None
        steps = min(max_length, dataset.get_output_length() - 1)
        
        X_samp = generate(
            model, X_init, steps, 
            temperature=temperature, 
            top_k=top_k_val, 
            do_sample=True
        ).to('cpu')
        
        # Decode generated names
        names = []
        for i in range(X_samp.size(0)):
            row = X_samp[i, 1:].tolist()  # Skip the first token (start letter or start token)
            # Find the end token (0) and crop there
            if 0 in row:
                row = row[:row.index(0)]
            # Decode the name
            name = dataset.decode(row)
            if name:
                # Prepend the start_with letter if specified
                if start_with:
                    full_name = start_with.lower() + name
                else:
                    full_name = name
                names.append(full_name)
        
        return names[:num_names] if names else ["Unable to generate names with current settings"]
    
    except Exception as e:
        return [f"Error generating names: {str(e)}"]


def load_help_md():
    """Load the NAME_GENERATION_README.md file and return its markdown text."""
    try:
        help_path = Path(__file__).parent / "NAME_GENERATION_README.md"
        text = help_path.read_text(encoding="utf-8")
        # strip surrounding triple-backtick fences if present
        if text.startswith("```"):
            parts = text.splitlines()
            # remove first fence line
            if parts and parts[0].startswith("```"):
                parts = parts[1:]
            # remove last fence line if present
            if parts and parts[-1].startswith("```"):
                parts = parts[:-1]
            text = "\n".join(parts)
        return text
    except Exception as e:
        return f"Error loading help file: {e}"

def main():
    # Use a centered column so header and button share the exact same alignment grid
    left, center, right = st.columns([1, 5, 1])
    with center:
        st.markdown('<div class="hero hero-top">', unsafe_allow_html=True)
        st.markdown('<h1 class="main-header">🏔️ Nepali Name Generator</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Transformer-based Neural Networks for Nepali Name Generation.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## ⚙️ Generation Settings")
        
        # Gender preference
        gender_preference = st.selectbox(
            "👥 Gender Preference",
            ["both", "male", "female"],
            index=0,
            help="Choose the type of names to generate"
        )
        
        # Number of names
        num_names = st.slider(
            "🔢 Number of Names",
            min_value=1,
            max_value=50,
            value=9,
            help="How many names to generate"
        )
        
        # Temperature (creativity)
        temperature = st.slider(
            "🎨 Creativity Level",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Lower values = more consistent, Higher values = more creative"
        )
        
        # Top-k sampling
        top_k = st.slider(
            "🎯 Focus Level",
            min_value=1,
            max_value=50,
            value=10,
            help="Lower values = more focused, Higher values = more diverse"
        )
        
        # Max length
        max_length = st.slider(
            "📏 Maximum Name Length",
            min_value=3,
            max_value=20,
            value=12,
            help="Maximum number of characters in generated names"
        )
        
        # Start with specific letters
        start_with = st.text_input(
            "🔤 Start Names With",
            value="",
            placeholder="e.g., 'A' or 'Ra'",
            help="Generate names starting with specific letters (optional)"
        )
        
        st.markdown("---")
        
        # Model information
        with st.expander("ℹ️ Model Information"):
            st.markdown("""
            This generator uses transformer-based neural networks trained on:
            - **6,212** male Nepali names (cleaned)
            - **2,575** female Nepali names (cleaned)
            - **Only a-z characters** for optimal training
            - Advanced attention mechanisms
            - Character-level language modeling
            
            **Data Cleaning:** All names have been preprocessed to use only lowercase 
            a-z characters, removing special characters, numbers, and mixed case for 
            consistent and stable training.
            """)

        # Help & Documentation - open full help in main area when clicked
        if st.button("📚 Open Help & Documentation"):
            st.session_state.show_help = True


    # Check if settings have changed
    current_settings = {
        'gender': gender_preference,
        'num_names': num_names,
        'temperature': temperature,
        'top_k': top_k,
        'max_length': max_length,
        'start_with': start_with
    }
    
    # Track if settings changed
    if 'previous_settings' not in st.session_state:
        st.session_state.previous_settings = current_settings
        st.session_state.settings_changed = False
    else:
        st.session_state.settings_changed = (st.session_state.previous_settings != current_settings)
        if st.session_state.settings_changed:
            st.session_state.previous_settings = current_settings

    # Generate button placed inside the same centered column, middle of a 3-col row
    with center:
        # Highlight if settings changed
        if hasattr(st.session_state, 'settings_changed') and st.session_state.settings_changed and hasattr(st.session_state, 'generated_names'):
            st.markdown("""<style>.stButton > button:first-child { animation: pulse 2s infinite !important; background: linear-gradient(45deg, #FF4444, #FF6B6B) !important; box-shadow: 0 6px 20px rgba(255,68,68,0.35) !important; }</style>""", unsafe_allow_html=True)

        btn_cols = st.columns([1,1,1])
        with btn_cols[1]:
            st.markdown('<div class="cta">', unsafe_allow_html=True)
            generate_clicked = st.button(
                "🚀 Generate Names" if not (hasattr(st.session_state, 'settings_changed') and st.session_state.settings_changed and hasattr(st.session_state, 'generated_names'))
                else "🔄 Regenerate with New Settings",
                key="generate",
                help="Click to generate beautiful Nepali names!"
            )
            # subtle instruction centered under the button
            if not hasattr(st.session_state, 'generated_names'):
                st.markdown('<div class="helper">Adjust settings in the sidebar and click <strong>Generate Names</strong>.</div>', unsafe_allow_html=True)
            elif hasattr(st.session_state, 'settings_changed') and st.session_state.settings_changed:
                st.markdown('<div class="helper" style="color: rgba(255,200,200,0.85);">⚠️ Settings changed — click <strong>Generate</strong>.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Main content area
    content_col = st.container()
    with content_col:
        # If user clicked the Help button, render the full README here with a close control
        if st.session_state.get('show_help', False):
            st.markdown("# 📚 Help & Documentation")
            # Render a Close control and ensure a single click fully closes help by
            # clearing the session flag and triggering an immediate rerun.
            if st.button("Close Help"):
                st.session_state.show_help = False
                # Force a rerun so the help section is removed in the same interaction
                try:
                    st.experimental_rerun()
                except Exception:
                    # If rerun isn't available for some Streamlit versions, fall back
                    # to normal behaviour — the help flag is already cleared.
                    pass

            # Load and display the help markdown. Keep the return so the rest of
            # the UI does not render while help is visible.
            help_md = load_help_md()
            st.markdown(help_md, unsafe_allow_html=False)
            # stop further UI rendering while help is visible
            return

        if generate_clicked:
            # Show progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner(""):
                # Load model with progress updates
                status_text.text("🔄 Loading AI model...")
                progress_bar.progress(25)
                
                model, dataset, model_info, error = load_model_and_data(gender_preference)
                
                if error:
                    st.error(f"❌ Error loading model: {error}")
                    st.info("💡 Please make sure you have trained models in the models/ directory")
                    progress_bar.empty()
                    status_text.empty()
                    return
                
                progress_bar.progress(50)
                status_text.text("🎨 Crafting beautiful names...")
                
                # Generate names
                config = Config(
                    sampling=SamplingConfig(
                        temperature=temperature,
                        top_k=top_k,
                        max_new_tokens=max_length,
                        num_samples=num_names
                    )
                )
                
                names = generate_names(
                    model, dataset, config, num_names, 
                    temperature, top_k, max_length, start_with
                )
                
                progress_bar.progress(100)
                status_text.text("✨ Names generated successfully!")
                
                # Store in session state
                st.session_state.generated_names = names
                st.session_state.generation_settings = {
                    'gender': gender_preference,
                    'temperature': temperature,
                    'top_k': top_k,
                    'max_length': max_length,
                    'start_with': start_with,
                    'model_info': model_info
                }
                # Mark that new names were just generated
                st.session_state.new_names_generated = True
                st.session_state.settings_changed = False  # Reset the settings changed flag
                
                # Clear progress indicators
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
        
        # Display generated names with animations
        if hasattr(st.session_state, 'generated_names'):
            st.markdown("## 🎯 Generated Names")
            
            # Get settings for name display
            settings = st.session_state.generation_settings
            
            # Display names with conditional animation
            names = st.session_state.generated_names
            cols_per_row = 3
            
            # Check if these are newly generated names
            is_new_generation = getattr(st.session_state, 'new_names_generated', False)
            animation_class = "name-reveal" if is_new_generation else "name-static"
            
            # Create a container for the names (max-width wrapper for readability)
            st.markdown('<div class="names-container">', unsafe_allow_html=True)
            
            for i in range(0, len(names), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(names):
                        name = names[i + j]
                        gender_class = "male-card" if settings['gender'] == 'male' else "female-card" if settings['gender'] == 'female' else "name-card"
                        col.markdown(f'<div class="name-card {gender_class} {animation_class}">{name}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Reset the new generation flag
            if is_new_generation:
                st.session_state.new_names_generated = False
            
            # Download button for names
            names_text = "\n".join(st.session_state.generated_names)
            st.download_button(
                label="📥 Download Names as Text File",
                data=names_text,
                file_name=f"nepali_names_{settings['gender']}_{len(names)}.txt",
                mime="text/plain"
            )
    
    # ...existing code...
    
    # # Footer
    # st.markdown("---")
    # st.markdown("""
    # <div style='text-align: center; color: #666; padding: 2rem;'>
    #     <p>🏔️ <strong>Nepali Name Generator</strong> - Powered by Transformer Neural Networks</p>
    #     <p>Built with ❤️ using Streamlit and PyTorch</p>
    # </div>
    # """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
