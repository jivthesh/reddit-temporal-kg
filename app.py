"""
app.py

Enterprise-grade SaaS dashboard style Streamlit Frontend for the Reddit Temporal Knowledge Graph.
Styled specifically to match a clean, light-mode, modern look with real-time database queries
and unified LLMClient completion tracing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import logging
import time
import re as _re
from typing import Dict, Any, List, Tuple, Optional


# Import backend clients
try:
    from config.settings import validate_settings, NEO4J_URI, WEAVIATE_URL
    from src.storage.graph_client import GraphClient
    from src.storage.vector_client import VectorClient
    from src.retrieval.hybrid_fusion import HybridFusion
    from src.llm.answer_generator import AnswerGenerator
    from src.llm.llm_client import LLMClient
    BACKEND_READY = True
except Exception as e:
    BACKEND_READY = False
    BACKEND_ERROR_MSG = str(e)
    logging.error(f"Backend modules not fully importable: {e}")

# ================= PAGE CONFIGURATION =================
st.set_page_config(
    page_title="Reddit Temporal Knowledge Graph",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS FOR SAAS LIGHT-MODE LOOK =================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Overrides (Force Light Mode Style) */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
        color: #1E293B !important;
    }
    
    /* Remove default Streamlit footer and main menu, keeping header for sidebar control */
    footer, #MainMenu {
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        background: none !important;
        -webkit-text-fill-color: initial !important;
        -webkit-background-clip: initial !important;
        margin-top: 0 !important;
    }
    
    h1 {
        font-size: 28px !important;
        letter-spacing: -0.5px !important;
    }
    
    .subtitle {
        color: #64748B;
        font-size: 14px;
        margin-top: -10px;
        margin-bottom: 24px;
    }

    /* Style Streamlit Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        color: #64748B !important;
        border-bottom: 2px solid transparent !important;
        background: transparent !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1B6EF3 !important;
        border-bottom-color: #1B6EF3 !important;
        font-weight: 600 !important;
    }
    
    /* SaaS Card Layout */
    .saas-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 2px !important;
        padding: 20px !important;
        box-shadow: none !important;
        margin-bottom: 20px !important;
    }
    
    /* Form Inputs */
    input[type="text"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #0F172A !important;
        border-radius: 2px !important;
    }
    
    /* Search success alert banner */
    .search-success {
        background-color: #ECFDF5;
        border: 1px solid #D1FAE5;
        border-radius: 2px;
        padding: 8px 16px;
        color: #065F46;
        font-size: 13px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* SaaS styled HTML tables */
    .saas-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        color: #334155;
        margin-top: 10px;
    }
    .saas-table th {
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid #E2E8F0;
        color: #64748B;
        font-weight: 500;
        background-color: #F8FAFC;
    }
    .saas-table td {
        padding: 12px 12px;
        border-bottom: 1px solid #F1F5F9;
        vertical-align: middle;
    }
    .saas-table tr:last-child td {
        border-bottom: none;
    }
    
    /* Styled badges */
    .table-badge {
        padding: 2px 8px;
        border-radius: 2px;
        font-size: 12px;
        font-weight: 600;
        text-align: center;
    }
    .badge-purple { background-color: #F3E8FF; color: #6B21A8; }
    .badge-green { background-color: #DCFCE7; color: #166534; }
    .badge-orange { background-color: #FFEDD5; color: #9A3412; }
    
    /* Buttons Custom Layout */
    div.stButton > button[data-testid="stBaseButton-primary"],
    div.stButton > button[kind="primary"] {
        background-color: #1B6EF3 !important;
        color: white !important;
        border: none !important;
        border-radius: 2px !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
        width: 100% !important;
    }
    div.stButton > button[data-testid="stBaseButton-primary"]:hover,
    div.stButton > button[kind="primary"]:hover {
        background-color: #1558C8 !important;
    }
    div.stButton > button[data-testid="stBaseButton-secondary"],
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 2px !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
        width: 100% !important;
    }
    div.stButton > button[data-testid="stBaseButton-secondary"]:hover,
    div.stButton > button[kind="secondary"]:hover {
        border-color: #94A3B8 !important;
        background-color: #F8FAFC !important;
    }
    
    /* High contrast text visibility overrides for Checkboxes & Metrics */
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] label span,
    [data-testid="stCheckbox"] p,
    [data-testid="stCheckbox"] span {
        color: #1E293B !important;
    }
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] span,
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] span {
        color: #1E293B !important;
    }
</style>

""", unsafe_allow_html=True)

# ================= BACKEND CLIENT CACHING =================
@st.cache_resource
def initialize_backend_clients():
    """Attempts to connect to Neo4j and Weaviate."""
    if not BACKEND_READY:
        return None, f"Import Error: {BACKEND_ERROR_MSG}"
    
    try:
        # Validate settings first
        validate_settings()
        
        # Connect to databases
        graph = GraphClient()
        vector = VectorClient()
        fusion = HybridFusion(graph, vector)
        answer_gen = AnswerGenerator()
        
        return {
            "graph": graph,
            "vector": vector,
            "fusion": fusion,
            "answer_gen": answer_gen
        }, None
    except Exception as e:
        return None, str(e)

# Initialize backend
clients, backend_error = initialize_backend_clients()

# ================= STATE MANAGEMENT FOR PRESETS & CLEAR =================
if "query_val" not in st.session_state:
    st.session_state["query_val"] = "How has sentiment around RAG changed over the past 6 months?"
if "search_cache" not in st.session_state:
    st.session_state["search_cache"] = None
if "trigger_search" not in st.session_state:
    st.session_state["trigger_search"] = False
# Track whether a modal/button (non-search) caused the last rerun
# so the pipeline is not auto-fired from a "View All" click

if "last_searched_query" not in st.session_state:
    st.session_state["last_searched_query"] = None

# Sync preset session state back to query input
if "selected_preset" in st.session_state:
    st.session_state["query_val"] = st.session_state["selected_preset"]
    st.session_state["trigger_search"] = True
    del st.session_state["selected_preset"]

# ================= SIDEBAR OPTIONS =================
with st.sidebar:
    st.markdown("<h3 style='margin-bottom: 20px; font-size: 18px; display: flex; align-items: center; gap: 8px;'>Settings</h3>", unsafe_allow_html=True)
    
    time_preset = st.selectbox(
        "Time Range",
        ["All Time", "Last 6 Months", "Last 3 Months", "Last Year"]
    )
    # Map to the label format parse_time_range_from_query understands
    _time_map = {
        "All Time":       None,
        "Last 6 Months":  "6_months",
        "Last 3 Months":  "3_months",
        "Last Year":      "1_year",
    }
    selected_time_range = _time_map[time_preset]

    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        ["Hybrid (Recommended)", "Graph Only (Neo4j)", "Vector Only (Weaviate)"],
        index=0
    )
    # Map UI label → internal query_type string used by HybridFusion
    _mode_map = {
        "Hybrid (Recommended)": "hybrid",
        "Graph Only (Neo4j)": "graph",
        "Vector Only (Weaviate)": "vector",
    }
    selected_query_type = _mode_map[retrieval_mode]

    top_k = st.slider("Top Results (K)", 3, 20, 10)
        
    active_llm_name = "LLM"
    if BACKEND_READY:
        try:
            from src.llm.answer_generator import llm_client
            active_llm_name = llm_client.provider.capitalize()
        except Exception:
            pass

# Stop app execution if connections are offline
if clients is None:
    st.title("🔍 Reddit Temporal Knowledge Graph")
    st.error(f"❌ Connection Failure: {backend_error}")
    st.warning("Please ensure your Docker containers (Neo4j and Weaviate) are active and running locally.")
    st.stop()

# ================= HEADER LAYOUT =================
st.markdown("<h1>Reddit Temporal Knowledge Graph</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Hybrid Semantic + Relational Retrieval System</p>", unsafe_allow_html=True)

# ================= TAB CONTROLS =================
tab_search, tab_analytics, tab_demos = st.tabs([
    "Search",
    "Analytics",
    "Demos"
])

# ================= HELPER FUNCTIONS TO RENDER TABLES =================

def _get_source_badge_style(source: str) -> Tuple[str, str, str]:
    """Returns the background color, text color, and label for a given source string."""
    src = (source or "hybrid").lower()
    if src.startswith("graph"):
        return "#F3E8FF", "#6B21A8", "Graph"
    if src.startswith("vector"):
        return "#DCFCE7", "#166534", "Vector"
    return "#FFEDD5", "#9A3412", "Hybrid"


def _clean_text(text: str) -> str:
    """Strip markdown formatting and leading punctuation from raw chunk text."""
    if not text:
        return ""
    # Remove markdown bold/italic markers (**text**, *text*)
    text = _re.sub(r'\*{1,2}', '', text)
    # Remove markdown heading markers
    text = _re.sub(r'^#+\s*', '', text, flags=_re.MULTILINE)
    # Strip leading punctuation and whitespace (e.g. ". However" → "However")
    text = _re.sub(r'^[\s.,;:!?\-–—]+', '', text)
    return text.strip()

def render_graph_table(results):
    rows = []
    for idx, r in enumerate(results, 1):
        title = r.get("title", "")
        if len(title) > 40:
            title = title[:38] + "..."
        subreddit = r.get("subreddit", "")
        sub_badge = f'<span class="table-badge badge-purple">r/{subreddit}</span>' if subreddit else ''
        topics = r.get("topics", [])
        topic_badges = "".join([f'<span class="table-badge" style="background-color: #E2E8F0; color: #475569; margin-right: 4px;">{t}</span>' for t in topics[:2]])
        score = f"{r.get('score', 0.0):.2f}"
        rows.append(
            f'<tr>'
            f'<td style="font-weight: 600; color: #1E293B;">{idx}</td>'
            f'<td style="font-weight: 500;">{title}<br>{topic_badges}</td>'
            f'<td>{sub_badge}</td>'
            f'<td><span style="font-family: monospace; font-weight: 500;">{score}</span></td>'
            f'</tr>'
        )
    return (
        f'<table class="saas-table">'
        f'<thead><tr>'
        f'<th style="width: 8%;">Rank</th>'
        f'<th style="width: 50%;">Post</th>'
        f'<th style="width: 24%;">Subreddit</th>'
        f'<th style="width: 18%;">Score</th>'
        f'</tr></thead>'
        f'<tbody>'
        f'{"".join(rows) if rows else "<tr><td colspan=4 style=\"text-align: center; color: #64748B;\">No results found</td></tr>"}'
        f'</tbody>'
        f'</table>'
    )

def render_vector_table(results):
    rows = []
    for idx, r in enumerate(results, 1):
        text = _clean_text(r.get("text", ""))
        if len(text) > 40:
            text = text[:38] + "..."
        
        # Display the actual vector similarity score if available, otherwise fallback to score
        similarity_score = r.get("similarity_score")
        if similarity_score is not None:
            score_str = f"{similarity_score:.4f}"
        else:
            score_str = f"{r.get('score', 0.0):.2f}"
            
        rows.append(
            f'<tr>'
            f'<td style="font-weight: 600; color: #1E293B;">{idx}</td>'
            f'<td style="font-weight: 500;">{text}</td>'
            f'<td><span style="font-family: monospace; font-weight: 500;">{score_str}</span></td>'
            f'</tr>'
        )
    return (
        f'<table class="saas-table">'
        f'<thead><tr>'
        f'<th style="width: 10%;">Rank</th>'
        f'<th style="width: 75%;">Content Preview</th>'
        f'<th style="width: 15%;">Similarity</th>'
        f'</tr></thead>'
        f'<tbody>'
        f'{"".join(rows) if rows else "<tr><td colspan=3 style=\"text-align: center; color: #64748B;\">No results found</td></tr>"}'
        f'</tbody>'
        f'</table>'
    )

def render_fused_cards(results):
    if not results:
        return '<div style="text-align: center; color: #64748B; padding: 20px; font-size: 13px;">No fused results found</div>'
    
    cards = []
    max_score = max([r.get("rrf_score", 0.01) for r in results]) if results else 1.0
    for idx, r in enumerate(results, 1):
        title = r.get("title", "") or r.get("id", f"item_{idx}")
        title = _clean_text(title)
        text = _clean_text(r.get("text", ""))
        text_preview = text[:120] + "..." if len(text) > 120 else text
        if text_preview == title:
            text_preview = ""

        src = r.get("source", "hybrid") or "hybrid"
        badge_bg, badge_color, badge_label = _get_source_badge_style(src)

        subreddit = r.get("subreddit", "")
        sub_html = f'<span style="font-size: 11px; padding: 1px 6px; border-radius: 2px; background-color: #EDE9FE; color: #5B21B6; font-weight: 500; margin-left: 6px;">r/{subreddit}</span>' if subreddit else ""

        score = r.get("rrf_score", 0.0)
        percentage = min(int((score / max_score) * 100), 100) if max_score > 0 else 0

        preview_html = ""
        if text_preview:
            preview_html = f'<div style="font-size: 12px; color: #475569; line-height: 1.4; margin-top: 6px;">{text_preview}</div>'

        cards.append(
            f'<div style="border: 1px solid #E2E8F0; border-radius: 2px; padding: 10px 12px; margin-bottom: 8px; background-color: #FAFBFC;">'
            f'<div style="display: flex; justify-content: space-between; align-items: flex-start;">'
            f'<div style="flex: 1; min-width: 0;">'
            f'<div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">'
            f'<span style="font-size: 12px; font-weight: 700; color: #94A3B8;">#{idx}</span>'
            f'<span style="font-size: 13px; font-weight: 600; color: #1E293B;">{title}</span>'
            f'<span style="font-size: 11px; padding: 1px 6px; border-radius: 2px; background-color: {badge_bg}; color: {badge_color}; font-weight: 600;">{badge_label}</span>'
            f'{sub_html}'
            f'</div>'
            f'{preview_html}'
            f'</div>'
            f'<div style="display: flex; align-items: center; margin-left: 10px; flex-shrink: 0;">'
            f'<span style="font-family: Inter, sans-serif; font-size: 11px; font-weight: 500; color: #64748B; margin-right: 4px;">RRF:</span>'
            f'<span style="font-family: monospace; font-size: 12px; font-weight: 600; color: #0F172A;">{score:.4f}</span>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
    return "".join(cards)

# ================= MODAL DIALOGS =================
@st.dialog("All Graph Search Results", width="large")
def show_graph_modal(results):
    st.markdown(render_graph_table(results), unsafe_allow_html=True)

@st.dialog("All Vector Search Results", width="large")
def show_vector_modal(results):
    st.markdown(render_vector_table(results), unsafe_allow_html=True)

@st.dialog("All Fused Search Results (RRF)", width="large")
def show_fused_modal(results):
    st.markdown(render_fused_cards(results), unsafe_allow_html=True)

def render_all_sources_html(results):
    items = []
    for idx, r in enumerate(results, 1):
        title = r.get("title", "") or r.get("id", f"item_{idx}")
        score = r.get("rrf_score", 0.0)
        subreddit = r.get("subreddit", "")
        time_period = r.get("time_period", "N/A")
        source = r.get("source", "vector")
        text = _clean_text(r.get("text", ""))
        
        text_snippet = text[:200] + "..." if len(text) > 200 else text
        
        meta_parts = []
        if subreddit:
            meta_parts.append(f'r/{subreddit}')
        if time_period and time_period != "N/A":
            meta_parts.append(f'Period: {time_period}')
        meta_parts.append(f'Source: {source.upper()}')
        meta_line = " · ".join(meta_parts)
        
        items.append(
            f'<div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 2px; padding: 12px; margin-bottom: 12px;">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
            f'<span style="font-size: 13px; font-weight: 600; color: #1E293B;">[{idx}] {title}</span>'
            f'<span style="font-size: 12px; padding: 2px 8px; border-radius: 2px; background-color: #E2E8F0; color: #475569; font-weight: 500;">RRF: {score:.4f}</span>'
            f'</div>'
            f'<div style="font-size: 12px; color: #64748B; margin-bottom: 8px;">'
            f'{meta_line}'
            f'</div>'
            f'<div style="font-size: 13px; color: #334155; font-style: italic; line-height: 1.4;">'
            f'"{text_snippet}"'
            f'</div>'
            f'</div>'
        )
    return "".join(items)

@st.dialog("All Sources", width="large")
def show_sources_modal(results):
    st.markdown(render_all_sources_html(results), unsafe_allow_html=True)

def _run_search_pipeline(
    query: str,
    top_k: int,
    query_type: str,
    time_range: Optional[str],
    clients: Dict[str, Any],
) -> None:
    """Executes the search pipeline, updating the UI progress and storing results in session state."""
    progress_placeholder = st.empty()
    
    def update_progress(step: int):
        lines = [
            "<div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 16px; border-radius: 4px; margin-bottom: 20px;'>",
            "<div style='font-weight: 600; color: #0F172A; margin-bottom: 12px; font-size: 14px;'>Processing Pipeline</div>"
        ]
        
        # Step 0
        if step == 0:
            lines.append("<div style='color: #64748B; font-size: 13px; margin-bottom: 6px;'><strong>Routing & Parsing</strong>: Categorizing query intent using LLM...</div>")
            lines.append("<div style='color: #94A3B8; font-size: 13px; margin-bottom: 6px;'><strong>Parallel DB Retrieval</strong>: Executing Neo4j Graph traversals & Weaviate Vector scans...</div>")
            lines.append("<div style='color: #94A3B8; font-size: 13px; margin-bottom: 6px;'><strong>Intelligent Fusion</strong>: Ranking combined documents via RRF...</div>")
            lines.append("<div style='color: #94A3B8; font-size: 13px;'><strong>LLM Answer Generation</strong>: Synthesizing final answer with source citations...</div>")
        # Step 1
        elif step == 1:
            lines.append("<div style='color: #16A34A; font-size: 13px; margin-bottom: 6px;'><strong>Routing & Parsing</strong> completed</div>")
            lines.append("<div style='color: #64748B; font-size: 13px; margin-bottom: 6px;'><strong>Parallel DB Retrieval & Intelligent Fusion</strong>: Executing DB queries & RRF ranking...</div>")
            lines.append("<div style='color: #94A3B8; font-size: 13px;'><strong>LLM Answer Generation</strong>: Synthesizing final answer with source citations...</div>")
        # Step 2
        elif step == 2:
            lines.append("<div style='color: #16A34A; font-size: 13px; margin-bottom: 6px;'><strong>Routing & Parsing</strong> completed</div>")
            lines.append("<div style='color: #16A34A; font-size: 13px; margin-bottom: 6px;'><strong>Parallel DB Retrieval & Intelligent Fusion</strong> completed</div>")
            lines.append("<div style='color: #64748B; font-size: 13px;'><strong>LLM Answer Generation</strong>: Synthesizing final answer with source citations...</div>")
            
        lines.append("</div>")
        progress_placeholder.markdown("\n".join(lines), unsafe_allow_html=True)

    update_progress(0)
    start_time = time.perf_counter()
    
    try:
        # 1. Route query
        with st.spinner("Classifying intent..."):
            from src.retrieval.query_router import QueryRouter
            router = QueryRouter()
            _ = router.route(query)
        
        update_progress(1)
        
        # 2. Retrieve & Fuse
        with st.spinner("Scanning databases..."):
            retrieval_output = clients["fusion"].retrieve(
                query,
                top_k=top_k,
                query_type_override=query_type,
                time_range_override=time_range,
            )
            
        update_progress(2)
        
        # 3. Generate Answer
        with st.spinner("Synthesizing answer..."):
            answer = clients["answer_gen"].generate(query, retrieval_output, append_sources=False)
        
        progress_placeholder.empty()
        elapsed_time = time.perf_counter() - start_time
        
        st.session_state["search_cache"] = {
            "query": query,
            "answer": answer,
            "retrieval_output": retrieval_output,
            "elapsed_time": elapsed_time
        }
        
    except Exception as ex:
        progress_placeholder.empty()
        st.error(f"❌ Pipeline Execution Failed: {ex}")
        st.warning("Ensure your API keys in the `.env` file are correct and have active quotas.")


def _render_tab_search(
    clients: Dict[str, Any],
    active_llm_name: str,
    top_k: int,
    selected_query_type: str,
    selected_time_range: Optional[str],
) -> None:
    """Renders the Search tab frontend elements, triggers, and results."""
    st.markdown("<p style='font-size: 14px; color: #475569; margin-bottom: 8px; font-weight: 500;'>Ask a question about Reddit discussions...</p>", unsafe_allow_html=True)
    
    col_input, col_search, col_clear = st.columns([6, 1.2, 1.2])
    
    with col_input:
        user_query_input = st.text_input(
            "Query Box",
            value=st.session_state["query_val"],
            label_visibility="collapsed",
            placeholder="How has sentiment around RAG changed over the past 6 months?"
        )
    with col_search:
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
    with col_clear:
        clear_btn = st.button("Clear", type="secondary", use_container_width=True)
        
    st.markdown("<p style='font-size: 13px; color: #64748B; margin-top: 6px; margin-bottom: 24px;'>Example: What are the main concerns about RAG systems?</p>", unsafe_allow_html=True)
    
    # Handle Clear Trigger
    if clear_btn:
        st.session_state["query_val"] = ""
        st.session_state["search_cache"] = None
        st.session_state["trigger_search"] = False
        st.rerun()
        
    # Determine search triggers.
    # IMPORTANT: Only fire a search when the user clicks the Search button
    # or when a preset is selected (trigger_search flag). The system does NOT auto-search
    # on text change because any button click (including "View All" modals)
    # causes a full Streamlit rerun — triggering a new search unintentionally.

    should_search = search_btn or st.session_state.get("trigger_search", False)

    # Execute query only when explicitly triggered
    if should_search:
        st.session_state["trigger_search"] = False
        st.session_state["last_searched_query"] = user_query_input
        st.write("---")
        _run_search_pipeline(user_query_input, top_k, selected_query_type, selected_time_range, clients)

    # Render cached search results if present
    if st.session_state["search_cache"] is not None:
        cache = st.session_state["search_cache"]
        query = cache["query"]
        answer = cache["answer"]
        retrieval_output = cache["retrieval_output"]
        elapsed_time = cache["elapsed_time"]
        
        st.write("---")
        
        success_banner_html = f"""
        <div class="search-success">
            Search completed in {elapsed_time:.2f} seconds
        </div>
        """
        st.markdown(success_banner_html, unsafe_allow_html=True)
        
        fused_results = retrieval_output.get("results", [])
        graph_results_display = [r for r in fused_results if r.get("source", "").startswith("graph")]
        vector_results_display = [r for r in fused_results if r.get("source", "").startswith("vector")]
        
        # Display Answer Card
        st.markdown(f"""
        <div class="saas-card" style="margin-top: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h4 style="margin: 0; font-size: 15px; color: #0F172A;">
                    Synthesized Answer ({active_llm_name})
                </h4>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(answer)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Compile sources list
        sources_items = []
        for idx, r in enumerate(fused_results[:5], 1):
            item_id = r.get("id", f"item_{idx}")
            score = r.get("rrf_score", 0.0)
            sources_items.append(
                f'<div style="font-size: 13px; color: #1E293B; margin-bottom: 12px; line-height: 1.4;">'
                f'<span style="color: #2563EB; font-weight: 600; margin-right: 4px;">[{idx}]</span>'
                f'<span style="font-family: monospace; font-size: 12px; background-color: #F1F5F9; padding: 2px 4px; border-radius: 2px;">{item_id}</span>'
                f'<span style="color: #64748B; font-size: 11px; margin-left: 4px;">(RRF: {score:.4f})</span>'
                f'</div>'
            )
        sources_list_html = "".join(sources_items)
        
        st.markdown(f"""<div class="saas-card" style="margin-bottom: 10px; margin-top: 20px;">
<h4 style="margin: 0 0 15px 0; font-size: 15px; color: #0F172A;">Sources</h4>
{sources_list_html if sources_list_html else "<div style='color:#64748B; font-size:12px;'>No sources listed</div>"}
<div style="margin-top: 10px;"></div>
</div>""", unsafe_allow_html=True)
        if st.button(f"View all {len(fused_results)} sources", key="btn_view_sources", use_container_width=True):
            show_sources_modal(fused_results)

        # Fused Results Cards
        st.markdown(f"""<div class="saas-card" style="margin-bottom: 10px; margin-top: 20px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
<h4 style="margin: 0; font-size: 15px; color: #0F172A;">
Fused Results (RRF)
</h4>
<span class="table-badge badge-orange">{len(fused_results)}</span>
</div>
{render_fused_cards(fused_results[:5])}
</div>""", unsafe_allow_html=True)
        if st.button("View all fused results", key="btn_view_f", use_container_width=True):
            show_fused_modal(fused_results)
            
        # Graph Tables
        st.markdown(f"""<div class="saas-card" style="margin-bottom: 10px; margin-top: 20px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
<h4 style="margin: 0; font-size: 15px; color: #0F172A;">
Graph Search Results
</h4>
<span class="table-badge badge-purple">{len(graph_results_display)}</span>
</div>
{render_graph_table(graph_results_display[:5])}
<div style="margin-top: 10px;"></div>
</div>""", unsafe_allow_html=True)
        if st.button("View all graph results", key="btn_view_g", use_container_width=True):
            show_graph_modal(graph_results_display)
            
        # Vector Tables
        st.markdown(f"""<div class="saas-card" style="margin-bottom: 10px; margin-top: 20px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
<h4 style="margin: 0; font-size: 15px; color: #0F172A;">
Vector Search Results
</h4>
<span class="table-badge badge-green">{len(vector_results_display)}</span>
</div>
{render_vector_table(vector_results_display[:5])}
<div style="margin-top: 10px;"></div>
</div>""", unsafe_allow_html=True)
        if st.button("View all vector results", key="btn_view_v", use_container_width=True):
            show_vector_modal(vector_results_display)


def _render_tab_analytics(clients: Dict[str, Any]) -> None:
    """Renders the Analytics tab with temporal and topic distribution charts."""
    st.markdown("### Real Database Temporal Analytics")
    st.markdown("Aggregated sentiment metrics and discussion volumes queried dynamically from your Neo4j instance.")

    # Data authenticity callout
    st.markdown("""
    <div style="background-color:#F0FDF4; border:1px solid #86EFAC; border-radius:2px;
                padding:12px 16px; margin-bottom:20px; display:flex; align-items:center; gap:12px;">
        <div>
            <div style="font-size:13px; font-weight:600; color:#166534;">
                Analysing REAL archived Reddit data</div>
            <div style="font-size:12px; color:#15803D; margin-top:2px;">
                Posts sourced from the <strong>Hugging Face / Pushshift</strong> public corpus &mdash;
                authentic discussions from r/MachineLearning, r/LanguageModels, r/artificial &amp; more.
                Zero synthetic generation.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown('<div class="saas-card"><h4>Sentiment Shifts Across Quarters</h4>', unsafe_allow_html=True)
        try:
            with clients["graph"].driver.session() as session:
                res = session.run("""
                    MATCH (p:Post)
                    WHERE p.time_period IS NOT NULL AND p.time_period <> ""
                    RETURN p.time_period as period, AVG(p.sentiment) as avg_sent
                    ORDER BY period
                """)
                records = [dict(r) for r in res]
                
            if records:
                periods = [r["period"] for r in records]
                sentiments = [r["avg_sent"] for r in records]
                
                chart_df = pd.DataFrame({
                    "Average Sentiment Score": sentiments
                }, index=periods)
                st.line_chart(chart_df)
            else:
                st.info("No temporal sentiment data found in database. Sentiment scores require post ingestion.")
        except Exception as e:
            st.error(f"Failed to query sentiment timeline: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_chart2:
        st.markdown('<div class="saas-card"><h4>Tracked Topic Distribution</h4>', unsafe_allow_html=True)
        try:
            with clients["graph"].driver.session() as session:
                res = session.run("""
                    MATCH (p:Post)-[:DISCUSSES]->(t:Topic)
                    RETURN t.name as topic, COUNT(p) as count
                    ORDER BY count DESC
                    LIMIT 8
                """)
                records = [dict(r) for r in res]
                
            if records:
                topics = [r["topic"] for r in records]
                counts = [r["count"] for r in records]
                
                vol_df = pd.DataFrame({
                    "Mention Count": counts
                }, index=topics)
                st.bar_chart(vol_df)
            else:
                st.info("No topic relationship records found. Check if graph schema is empty.")
        except Exception as e:
            st.error(f"Failed to query topic distribution: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
## Temporal Analysis Summary

**Dataset Scope:**
- **Duration**: 12 months (Q2 2025 – Q2 2026)
- **Posts**: 198 authentic Reddit discussions
- **Subreddits**: MachineLearning, LanguageModels, LocalLLaMA, mlops, datascience

**Key Findings:**
- **Sentiment Trend**: 0.769 (Q2 2025) → 0.665 (Q2 2026)
- **Topic Shift**: Speculative (early) → Practical (recent)
- **Community Focus**: Moved from "Can open-source compete?" to "How do we deploy effectively?"
""")


def _render_tab_demos() -> None:
    """Renders the pre-built demo query buttons."""
    st.markdown("### Pre-Built Demo Queries")
    st.markdown("Click on any preset query below to instantly populate the search input box:")
    
    demos = [
        {
            "title": "Semantic Query (Vector-Dominant)",
            "question": "What are the main concerns and criticisms about RAG systems?",
            "desc": "Pure semantic search - finds thematically similar content"
        },
        {
            "title": "Relationship Query (Graph-Dominant)",
            "question": "Who are the most active voices discussing AI safety on Reddit?",
            "desc": "Graph traversal - finds highly connected entities"
        },
        {
            "title": "Hybrid Query",
            "question": "How have community discussions about open-source LLMs shifted recently?",
            "desc": "Needs both: semantic + relationship understanding"
        },
        {
            "title": "Temporal Query",
            "question": "How has sentiment around RAG changed in the last 6 months?",
            "desc": "Time-series comparison across periods"
        }
    ]
    
    for i, demo in enumerate(demos, 1):
        with st.expander(demo['title']):
            st.write(f"**Question:** {demo['question']}")
            st.write(f"**Objective:** {demo['desc']}")
            
            if st.button(f"Load Demo Query {i}", key=f"demo_{i}"):
                st.session_state["selected_preset"] = demo["question"]
                st.rerun()
    # ================= TAB 1: SEARCH PAGE =================
with tab_search:
    _render_tab_search(
        clients=clients,
        active_llm_name=active_llm_name,
        top_k=top_k,
        selected_query_type=selected_query_type,
        selected_time_range=selected_time_range,
    )

# ================= TAB 2: TEMPORAL ANALYTICS =================
with tab_analytics:
    _render_tab_analytics(clients=clients)

# ================= TAB 3: HOW IT WORKS & DEMOS =================
with tab_demos:
    _render_tab_demos()

# ================= FOOTER =================
st.divider()
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #64748B;">
    <div>System Architecture: Neo4j (Graph) + Weaviate (Vectors) + {active_llm_name} (LLM) + Streamlit (Frontend)</div>
    <div style="display: flex; align-items: center; gap: 6px;">
        <span style="height: 6px; width: 6px; background-color: #10B981; border-radius: 50%; display: inline-block;"></span>
        Last updated: Q1 2026
    </div>
</div>
""", unsafe_allow_html=True)
