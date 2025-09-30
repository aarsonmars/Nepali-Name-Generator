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
import importlib.util

try:
    from nepali_translator.translator import Converter as PackageConverter  # type: ignore
except Exception:  # pragma: no cover - handled gracefully if dependency missing
    PackageConverter = None

_converter_instance = None
_converter_error = False


def _load_nepali_converter_via_files():
    """Attempt to load the converter by manually executing the package modules."""
    spec = importlib.util.find_spec("nepali_translator")
    if not spec or not spec.origin:
        return None
    package_dir = Path(spec.origin).parent
    translator_path = package_dir / "translator.py"
    mappings_path = package_dir / "mappings.py"
    if not translator_path.exists() or not mappings_path.exists():
        return None

    # Load mappings module under the expected top-level name used in translator.py
    spec_m = importlib.util.spec_from_file_location(
        "mappings", mappings_path
    )
    if not spec_m or not spec_m.loader:
        return None
    mappings_module = importlib.util.module_from_spec(spec_m)
    sys.modules.setdefault("mappings", mappings_module)
    spec_m.loader.exec_module(mappings_module)  # type: ignore[attr-defined]

    if not (package_dir / "words_maps.txt").exists():
        # Gracefully degrade when the published package misses its word map resource.
        def _empty_word_maps():
            return {}

        setattr(mappings_module, "get_word_maps", _empty_word_maps)

    # Load translator module and instantiate the Converter class
    spec_t = importlib.util.spec_from_file_location(
        "nepali_translator_runtime", translator_path
    )
    if not spec_t or not spec_t.loader:
        return None
    translator_module = importlib.util.module_from_spec(spec_t)
    spec_t.loader.exec_module(translator_module)  # type: ignore[attr-defined]

    converter_cls = getattr(translator_module, "Converter", None)
    if converter_cls is None:
        return None
    return converter_cls()


def get_nepali_converter():
    """Return a singleton converter instance, loading fallbacks if necessary."""
    global _converter_instance, _converter_error

    if _converter_instance is not None:
        return _converter_instance
    if _converter_error:
        return None

    try:
        if PackageConverter is not None:
            _converter_instance = PackageConverter()
            return _converter_instance

        fallback_converter = _load_nepali_converter_via_files()
        if fallback_converter is not None:
            _converter_instance = fallback_converter
            return _converter_instance
    except Exception:
        pass

    _converter_error = True
    return None

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import project modules
from config import Config, DataConfig, TrainingConfig, SamplingConfig, SystemConfig
from models import Transformer
from data import create_datasets
from utils import generate

# Page configuration
st.set_page_config(
    page_title="Nepali Name Generator",
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
    :root {
        --text-color: #f5f7fa;
        --subtitle-color: rgba(245,247,250,0.9);
        --helper-color: rgba(255,255,255,0.72);
        --info-panel-bg: linear-gradient(180deg, rgba(17,24,39,0.95), rgba(15,23,42,0.85));
        --info-panel-text-color: rgba(255,255,255,0.95);
        --info-panel-muted-color: rgba(255,255,255,0.74);
        --info-panel-border-color: rgba(255,255,255,0.04);
        --info-panel-shadow: 0 8px 30px rgba(2,6,23,0.6);
        --progress-bar-bg: rgba(255,255,255,0.08);
        --progress-fill-bg: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        --heading-text-shadow: 0 1px 2px rgba(0,0,0,0.45);
        --name-card-text-shadow: 0 1px 2px rgba(0,0,0,0.35);
        --card-border-color: rgba(255,255,255,0.08);
        --card-box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        --card-hover-box-shadow: 0 18px 50px rgba(0,0,0,0.45);
    --divider-gradient: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.14), rgba(255,255,255,0));
        --surface-card-bg: rgba(17,25,40,0.72);
        --surface-card-border: rgba(255,255,255,0.08);
        --surface-card-shadow: 0 18px 45px rgba(2,6,23,0.45);
        --surface-card-hover: 0 22px 55px rgba(2,6,23,0.55);
        --badge-bg: rgba(78,204,196,0.15);
        --badge-text: #4ECDC4;
        --empty-state-bg: rgba(15,23,42,0.75);
        --empty-state-border: rgba(78,204,196,0.3);
        --tip-bg: rgba(15,23,42,0.72);
        --tip-border: rgba(255,255,255,0.04);
    }

    body[data-theme="light"] {
        --text-color: #0f1723;
        --subtitle-color: rgba(15,23,35,0.78);
        --helper-color: rgba(15,23,35,0.65);
        --info-panel-bg: linear-gradient(180deg, #f7faff, #e5eefc);
        --info-panel-text-color: #102a43;
        --info-panel-muted-color: rgba(16,42,67,0.7);
        --info-panel-border-color: rgba(16,42,67,0.12);
        --info-panel-shadow: 0 12px 35px rgba(15,23,35,0.12);
        --progress-bar-bg: rgba(15,23,35,0.12);
        --progress-fill-bg: linear-gradient(90deg, #2563eb, #10b981);
        --heading-text-shadow: 0 1px 1px rgba(255,255,255,0.6);
        --name-card-text-shadow: none;
        --card-border-color: rgba(15,23,35,0.12);
        --card-box-shadow: 0 6px 20px rgba(15,23,35,0.12);
        --card-hover-box-shadow: 0 14px 35px rgba(15,23,35,0.18);
    --divider-gradient: linear-gradient(90deg, rgba(15,23,35,0), rgba(15,23,35,0.18), rgba(15,23,35,0));
        --surface-card-bg: #ffffff;
        --surface-card-border: rgba(15,23,35,0.08);
        --surface-card-shadow: 0 14px 32px rgba(15,23,35,0.12);
        --surface-card-hover: 0 18px 40px rgba(15,23,35,0.16);
        --badge-bg: rgba(37,99,235,0.12);
        --badge-text: #2563eb;
        --empty-state-bg: linear-gradient(180deg, #f8fbff, #eef3ff);
        --empty-state-border: rgba(37,99,235,0.18);
        --tip-bg: #f7faff;
        --tip-border: rgba(15,23,35,0.1);
    }

    /* Base typography and theme-aware defaults */
    body, .stApp {
        font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        color: var(--text-color);
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
        color: var(--subtitle-color);
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
        color: var(--helper-color);
        font-size: 0.95rem;
        line-height: 1.35;
        max-width: 640px;
    }

    /* Overview metrics and feature highlights */

    /* Name tiles - premium glass gradients */
    .name-card {
        position: relative;
        background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid var(--card-border-color);
        color: #f7fbff;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        margin: 0.5rem 0.35rem;
        text-align: center;
        font-size: 1.08rem;
        font-weight: 700;
        box-shadow: var(--card-box-shadow);
        text-shadow: var(--name-card-text-shadow);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        transition: transform 0.22s ease, box-shadow 0.22s ease, filter 0.22s ease;
        min-height: 52px;
    }

    .male-card { background: linear-gradient(135deg, #00D2FF 0%, #3A47D5 100%); }
    .female-card { background: linear-gradient(135deg, #FF6CAB 0%, #7366FF 100%); }
    .name-card:hover { transform: translateY(-6px) scale(1.015); box-shadow: var(--card-hover-box-shadow); filter: saturate(1.05); }

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
        position: relative !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.55rem !important;
        padding: 0.85rem 2.2rem !important;
        border-radius: 16px !important;
        font-weight: 700 !important;
        letter-spacing: 0.28px !important;
        text-transform: uppercase;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(255,255,255,0.05) !important;
        color: var(--text-color) !important;
        box-shadow: 0 12px 28px rgba(2,6,23,0.28) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
        overflow: hidden !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 18px 38px rgba(2,6,23,0.32) !important;
        background: rgba(255,255,255,0.09) !important;
    }

    .stButton > button:active {
        transform: translateY(0) scale(0.98) !important;
        box-shadow: 0 10px 22px rgba(2,6,23,0.25) !important;
    }

    .stButton > button:focus-visible {
        outline: 2px solid rgba(78,204,196,0.6) !important;
        outline-offset: 2px !important;
    }

    .generate-button-wrapper .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        background-size: 200% 200% !important;
        color: #ffffff !important;
        padding: 1.5rem 4.2rem !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 30px !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 25px 60px rgba(102, 126, 234, 0.4), 0 0 0 1px rgba(255,255,255,0.1) inset !important;
        animation: gradientFlow 8s ease infinite, buttonGlow 3s ease-in-out infinite;
        isolation: isolate;
        position: relative !important;
        overflow: hidden !important;
        text-transform: none !important;
        min-width: 280px !important;
    }

    .generate-button-wrapper .stButton > button p {
        margin: 0 !important;
        text-transform: none !important;
        letter-spacing: 0.3px;
        font-weight: 800;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        font-size: inherit;
    }

    .generate-button-wrapper .stButton > button::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.8s ease;
    }

    .generate-button-wrapper .stButton > button::after {
        content: "";
        position: absolute;
        inset: -8px;
        border-radius: 28px;
        background: radial-gradient(circle at top, rgba(255,255,255,0.55), rgba(255,255,255,0.05));
        opacity: 0.45;
        filter: blur(18px);
        z-index: -1;
        transition: opacity 0.32s ease, filter 0.32s ease;
        animation: buttonAura 4.6s ease-in-out infinite;
    }

    .generate-button-wrapper .stButton > button:hover {
        transform: translateY(-10px) scale(1.06) !important;
        box-shadow: 0 40px 80px rgba(102, 126, 234, 0.5) !important;
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
    }

    .generate-button-wrapper .stButton > button:hover::before {
        left: 100%;
    }

    .generate-button-wrapper .stButton > button:hover::after {
        opacity: 0.8;
        filter: blur(25px);
    }

    .generate-button-wrapper .stButton > button:active {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4) !important;
    }

    /* Generate area styling */
    .generate-card {
        display: flex;
        gap: 1rem;
        align-items: center;
    }

    .info-panel {
        background: var(--info-panel-bg);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: var(--info-panel-text-color);
        box-shadow: var(--info-panel-shadow);
        border: 1px solid var(--info-panel-border-color);
        font-weight: 500;
        line-height: 1.35;
    }

    .info-panel .muted { color: var(--info-panel-muted-color); font-weight: 400; font-size: 0.95rem; }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin: 1rem 0 1.5rem;
    }

    .summary-card {
        background: var(--surface-card-bg);
        border: 1px solid var(--surface-card-border);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: var(--surface-card-shadow);
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .summary-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--surface-card-hover);
    }

    .summary-header {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.16rem;
        color: rgba(255,255,255,0.6);
    }

    body[data-theme="light"] .summary-header { color: rgba(15,23,35,0.6); }

    .summary-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: inherit;
    }

    .summary-caption {
        font-size: 0.9rem;
        color: var(--helper-color);
        line-height: 1.35;
    }

    .empty-state {
        background: var(--empty-state-bg);
        border: 1px solid var(--empty-state-border);
        border-radius: 18px;
        padding: 2.4rem 2rem;
        text-align: center;
        max-width: 720px;
        margin: 1.4rem auto 0;
        box-shadow: var(--surface-card-shadow);
        color: var(--text-color);
    }

    .empty-state h3 {
        font-size: 1.6rem;
        margin-bottom: 0.6rem;
    }

    .empty-state p {
        font-size: 0.98rem;
        color: var(--helper-color);
        margin-bottom: 0;
    }

    .tips-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1rem;
        margin: 1rem 0 0;
    }

    .tip-card {
        background: var(--tip-bg);
        border: 1px solid var(--tip-border);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        box-shadow: var(--surface-card-shadow);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .tip-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--surface-card-hover);
    }

    .tip-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 1.05rem;
        font-weight: 700;
    }

    .tip-card p {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.45;
        color: var(--helper-color);
    }

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
        text-shadow: var(--heading-text-shadow);
        margin: 0.25rem 0;
    }

    /* Animations and micro-interactions (kept) */
    .skeleton { background: linear-gradient(90deg, #e6e6e6 25%, #dddddd 50%, #e6e6e6 75%); background-size: 200% 100%; animation: loading 1.5s infinite; border-radius: 8px; }
    @keyframes loading { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
    @keyframes gradientFlow { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    @keyframes buttonAura { 0%, 100% { opacity: 0.35; transform: scale(0.96); } 50% { opacity: 0.8; transform: scale(1.06); } }
    @keyframes buttonGlow { 0%, 100% { box-shadow: 0 25px 60px rgba(102, 126, 234, 0.4); } 50% { box-shadow: 0 30px 70px rgba(102, 126, 234, 0.6); } }

    /* Name cards: center text and ensure consistent vertical alignment */
    .name-card { transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1); cursor: pointer; display: flex; align-items: center; justify-content: center; min-height: 52px; }
    .name-card:hover { transform: translateY(-6px) scale(1.02); box-shadow: var(--card-hover-box-shadow); }

    .typing-animation { border-right: 2px solid #FF6B6B; animation: blink 1s infinite; white-space: nowrap; overflow: hidden; }
    @keyframes blink { 0%,50% { border-color: #FF6B6B; } 51%,100% { border-color: transparent; } }

    .progress-bar { width: 100%; height: 8px; background-color: var(--progress-bar-bg); border-radius: 4px; overflow: hidden; margin: 1rem 0; }
    .progress-fill { height: 100%; background: var(--progress-fill-bg); border-radius: 4px; animation: progress 2s ease-in-out; }
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
    .section-divider { height: 1px; width: 92%; max-width: 900px; margin: 1rem auto 1.6rem; background: var(--divider-gradient); border-radius: 999px; }

    /* Nudge first markdown container up if Streamlit still applies spacing */
    div[data-testid="stMarkdownContainer"]:first-of-type {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* keep JS for typing/reveal intact (inlined) */
    body[data-theme="light"] .name-card { text-shadow: none; }
    body[data-theme="light"] .stats-container,
    body[data-theme="light"] .generation-info,
    body[data-theme="light"] .sidebar-content {
        box-shadow: 0 8px 24px rgba(15,23,35,0.1);
    }
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


def format_roman_name(name: str) -> str:
    """Format the Romanized name for display (title case each word)."""
    if not name:
        return ""
    return " ".join(part.capitalize() for part in name.split())


def transliterate_to_nepali(name: str) -> str:
    """Convert a Romanized Nepali name to Unicode Nepali using the translator package."""
    if not name:
        return ""
    try:
        converter = get_nepali_converter()
        if converter is None:
            return ""
        name_lc = name.lower().strip()
        if not name_lc:
            return ""
        vowels = "eiou"
        last_char = name_lc[-1]
        base = name_lc
        if last_char.isalpha() and last_char not in vowels:
            base = name_lc + "a"
        return converter.convert(base)
    except Exception:
        return ""


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
        st.markdown('<h1 class="main-header">Nepali Name Generator</h1>', unsafe_allow_html=True)
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


    # Check if settings have changed since the last generation
    if 'previous_settings' in st.session_state:
        st.session_state.pop('previous_settings', None)

    current_settings = {
        'gender': gender_preference,
        'num_names': num_names,
        'temperature': temperature,
        'top_k': top_k,
        'max_length': max_length,
        'start_with': start_with
    }

    last_generated_settings = st.session_state.get('last_generated_settings')
    settings_changed = False
    if last_generated_settings is not None:
        settings_changed = current_settings != last_generated_settings

    st.session_state.settings_changed = settings_changed

    # Generate button placed inside the same centered column, middle of a 3-col row
    with center:
        # Highlight if settings changed
        if settings_changed and st.session_state.get('generated_names'):
            st.markdown("""
            <style>
            .generate-button-wrapper .stButton > button {
                animation: gradientFlow 8s ease infinite, pulse 2s infinite !important;
                background: linear-gradient(130deg, #FF445A, #FF6B6B, #FF8AAE) !important;
                background-size: 240% 240% !important;
                box-shadow: 0 32px 75px rgba(255, 107, 107, 0.38) !important;
                color: #35060E !important;
            }
            </style>
            """, unsafe_allow_html=True)

        btn_cols = st.columns([1,1,1])
        with btn_cols[1]:
            st.markdown('<div class="cta generate-button-wrapper">', unsafe_allow_html=True)
            generate_clicked = st.button(
                "✨ Generate Names" if not (settings_changed and st.session_state.get('generated_names'))
                else "🔄 Regenerate with New Settings",
                key="generate",
                help="Click to generate beautiful Nepali names with AI!"
            )
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
                
                formatted_entries = []
                for raw_name in names:
                    clean_name = raw_name.strip()
                    roman_display = format_roman_name(clean_name)
                    nepali_text = transliterate_to_nepali(clean_name)
                    formatted_entries.append({
                        "roman": roman_display,
                        "nepali": nepali_text,
                        "raw": clean_name
                    })

                # Store in session state
                st.session_state.generated_entries = formatted_entries
                st.session_state.generated_names = [entry["roman"] for entry in formatted_entries]
                st.session_state.generation_settings = {
                    'gender': gender_preference,
                    'num_names': num_names,
                    'temperature': temperature,
                    'top_k': top_k,
                    'max_length': max_length,
                    'start_with': start_with,
                    'model_info': model_info
                }
                # Mark that new names were just generated
                st.session_state.new_names_generated = True
                st.session_state.last_generated_settings = current_settings.copy()
                st.session_state.settings_changed = False  # Reset the settings changed flag
                
                # Clear progress indicators
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
        
        # Display generated names, summaries, or onboarding cues
        generated_entries = st.session_state.get('generated_entries')
        if not generated_entries and hasattr(st.session_state, 'generated_names'):
            fallback_names = st.session_state.generated_names
            generated_entries = []
            for fallback_name in fallback_names:
                clean_fallback = fallback_name.strip()
                generated_entries.append({
                    "roman": format_roman_name(clean_fallback),
                    "nepali": transliterate_to_nepali(clean_fallback),
                    "raw": clean_fallback
                })
            st.session_state.generated_entries = generated_entries

        if generated_entries:
            settings = st.session_state.generation_settings

            # Determine animation class for cards
            is_new_generation = getattr(st.session_state, 'new_names_generated', False)
            animation_class = "name-reveal" if is_new_generation else "name-static"
            start_letters = settings['start_with'].strip()

            st.markdown("## 🎯 Generated Names")
            st.markdown('<div class="names-container">', unsafe_allow_html=True)

            cols_per_row = 3
            for i in range(0, len(generated_entries), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(generated_entries):
                        entry = generated_entries[i + j]
                        card_text = entry["roman"]
                        if entry["nepali"]:
                            card_text = f"{entry['roman']} | {entry['nepali']}"
                        gender_class = (
                            "male-card" if settings['gender'] == 'male'
                            else "female-card" if settings['gender'] == 'female'
                            else "name-card"
                        )
                        col.markdown(
                            f'<div class="name-card {gender_class} {animation_class}">{card_text}</div>',
                            unsafe_allow_html=True
                        )

            st.markdown('</div>', unsafe_allow_html=True)

            download_lines = [
                f"{entry['roman']} | {entry['nepali']}" if entry['nepali'] else entry['roman']
                for entry in generated_entries
            ]
            names_text = "\n".join(download_lines)
            st.download_button(
                label="📥 Download Names as Text File",
                data=names_text,
                file_name=f"nepali_names_{settings['gender']}_{len(generated_entries)}.txt",
                mime="text/plain"
            )

            gender_label = {
                "both": "Blended",
                "male": "Masculine",
                "female": "Feminine"
            }.get(settings['gender'], settings['gender'].title())

            start_value = start_letters.upper() if start_letters else "Any"

            model_info = settings.get('model_info') or {}
            parameter_count = model_info.get('parameters')
            block_size = model_info.get('block_size')
            vocab_size = model_info.get('vocab_size')
            device = model_info.get('device', 'cpu')

            summary_items = [
                {"header": "Gender focus", "value": gender_label, "caption": "Model selection"},
                {"header": "Creativity", "value": f"{settings['temperature']:.1f}", "caption": "Temperature"},
                {"header": "Focus", "value": settings['top_k'], "caption": "Top-k sampling"},
                {"header": "Max length", "value": settings['max_length'], "caption": "Characters"},
                {"header": "Starting letters", "value": start_value, "caption": "Prefix constraint"},
            ]

            if vocab_size:
                summary_items.append({"header": "Vocabulary", "value": f"{vocab_size:,}", "caption": "Characters modeled"})
            if block_size:
                summary_items.append({"header": "Context window", "value": f"{block_size:,}", "caption": "Model look-back"})
            if parameter_count:
                summary_items.append({"header": "Parameters", "value": f"{parameter_count:,}", "caption": "Trainable weights"})
            if device:
                summary_items.append({"header": "Device", "value": device.upper(), "caption": "Inference"})

            summary_cards_html = "".join(
                (
                    '<div class="summary-card">'
                    f'<div class="summary-header">{item["header"]}</div>'
                    f'<div class="summary-value">{item["value"]}</div>'
                    f'<div class="summary-caption">{item["caption"]}</div>'
                    '</div>'
                )
                for item in summary_items
            )

            with st.expander("Generation summary", expanded=False):
                st.markdown('<div class="summary-grid">' + summary_cards_html + '</div>', unsafe_allow_html=True)
                st.markdown(
                    """
                    <p style="font-size:0.95rem; line-height:1.5; margin-top:0.5rem;">
                        These settings shape the character-level Transformer to balance tradition and novelty. 
                        Try lowering <strong>Creativity</strong> for classic spellings, or increasing <strong>Focus</strong> for broader exploration.
                    </p>
                    """,
                    unsafe_allow_html=True
                )

            tips = [
                {
                    "title": "Blend heritage with freshness",
                    "body": "Keep creativity around 0.8–1.1 for modern-yet-recognizable name suggestions."
                },
                {
                    "title": "Guide pronunciations",
                    "body": "Use starting letters like <strong>Pr</strong>, <strong>Ksh</strong>, or <strong>Mi</strong> to favor certain phonetics."
                },
                {
                    "title": "Export & shortlist",
                    "body": "Download generated names, group them by vibe, and share with friends or family for quick feedback."
                }
            ]

            if settings['temperature'] > 1.3:
                tips.insert(0, {
                    "title": "Too wild? Dial it back",
                    "body": "Lower the creativity slider towards 0.9 to reduce unexpected syllable combinations."
                })
            elif settings['temperature'] < 0.7:
                tips.insert(0, {
                    "title": "Add variety",
                    "body": "Increase creativity above 1.0 to unlock more adventurous name blends."
                })

            if settings['top_k'] < 6:
                tips.append({
                    "title": "Widen the search",
                    "body": "Raise the focus slider to sample from a broader set of likely next characters."
                })

            if start_letters:
                tips.append({
                    "title": "Try multiple prefixes",
                    "body": "Rotate through different starting letters to discover families of related names."
                })

            tips_html = "".join(
                (
                    '<div class="tip-card">'
                    f'<h4>{tip["title"]}</h4>'
                    f'<p>{tip["body"]}</p>'
                    '</div>'
                )
                for tip in tips
            )

            with st.expander("Tips & inspiration", expanded=False):
                st.markdown('<div class="tips-grid">' + tips_html + '</div>', unsafe_allow_html=True)

            if is_new_generation:
                st.session_state.new_names_generated = False

        else:
            st.markdown(
                """
                <div class="empty-state">
                    <h3>Ready to craft distinctive Nepali names?</h3>
                    <p>Pick a gender focus, tweak creativity and focus, then click <strong>Generate Names</strong> to begin.</p>
                    <ul style="margin-top:1rem; text-align:left; line-height:1.6;">
                        <li>Switch between masculine, feminine, or blended models instantly.</li>
                        <li>Tune creativity and focus to balance tradition with fresh spellings.</li>
                        <li>Download your shortlist and share it with family or friends.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
    #     <p>🏔️ <strong>Nepali Name Generator</strong> - Powered by Transformer Neural Networks</p>
    #     <p>Built with ❤️ using Streamlit and PyTorch</p>
    # </div>
    # """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
