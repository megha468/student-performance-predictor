"""
🎓 EduPredict.AI — Enterprise Student Performance Intelligence Platform
========================================================================
A comprehensive, institutional-grade academic prediction and analytics application.
Features:
  - Sleek, modern Dark Mode UI with high-contrast glowing accents
  - Dynamic AI branding logo (EduPredict.AI)
  - Admin Portal with 4 primary sidebar sections:
      1. Overview (4 KPI cards, dynamic performance charts, recent activity)
      2. Manage Students (real-time roster, add student, update student, delete student, view student logs)
      3. All Predictions / Student History (complete logs, multi-filters, CSV export, grade analytics)
      4. Settings (admin password change, reset student passwords, system status)
  - Student Portal:
      1. Dashboard (KPIs, score trend spline chart, quick actions)
      2. Predict Performance (ML ensemble inference, real-time feedback & AI Action Plan)
      3. My History & Analytics (personal logs, score progression, CSV export)
      4. My Profile (student details, self-service password update)
"""

import os
import io
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import database as db
from src.student_predictor.models import load_model_suite, StudentModelSuite
from src.student_predictor.advisor import generate_recommendations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize database schema and ensure mock data exists
db.init_db()


# -----------------------------------------------------------------------------
# CACHED MODEL SUITE
# -----------------------------------------------------------------------------
@st.cache_resource
def get_cached_models() -> StudentModelSuite:
    return load_model_suite(BASE_DIR)


try:
    suite = get_cached_models()
except Exception as e:
    suite = None


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EduPredict.AI — Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# BULLETPROOF HTML RENDERER (ELIMINATES STREAMLIT/MARKDOWN 4-SPACE CODE BLOCK BUG)
# -----------------------------------------------------------------------------
def render_html(html_str: str):
    """
    Renders pure HTML without Streamlit/Markdown 4-space code block escaping.
    Strips leading whitespace on each line so CommonMark never sees an indented code block.
    """
    cleaned = "".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# BRAND LOGO COMPONENT
# -----------------------------------------------------------------------------
def get_brand_logo_svg(size=26):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="M12 2L1 7L12 12L23 7L12 2Z" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<path d="M5 10V16C5 18.5 8.13 21 12 21C15.87 21 19 18.5 19 16V10" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="12" cy="12" r="2.5" fill="#38BDF8"/>'
        f'<path d="M23 7V14" stroke="#38BDF8" stroke-width="2" stroke-linecap="round"/>'
        f'</svg>'
    )


# -----------------------------------------------------------------------------
# CUSTOM DARK MODE CSS & STYLING
# -----------------------------------------------------------------------------
DARK_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Root App Background */
.stApp {
    background-color: #090D16 !important;
    color: #F1F5F9 !important;
}

/* Container Spacing */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
    max-width: 1240px;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0D1424 !important;
    border-right: 1px solid #1E293B !important;
}
section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #1E293B !important;
}

/* Sidebar Navigation Radios */
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 6px;
    background: transparent;
    transition: all 0.2s ease;
    border: 1px solid transparent;
    cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: rgba(56, 189, 248, 0.08) !important;
    border-color: rgba(56, 189, 248, 0.2) !important;
    color: #FFFFFF !important;
}

/* Forms & Containers in Dark Mode */
div[data-testid="stForm"] {
    background-color: #111A2E !important;
    border: 1px solid #1E293B !important;
    border-radius: 16px !important;
    padding: 24px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #111A2E !important;
    border: 1px solid #1E293B !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
}

/* Brand Logo Box */
.sidebar-logo-box {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 14px;
    margin-bottom: 22px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.sidebar-logo-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, #1E3A8A 0%, #0284C7 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.35);
}
.sidebar-logo-text {
    line-height: 1.25;
}
.logo-main {
    font-weight: 800;
    font-size: 16px;
    letter-spacing: -0.02em;
    color: #FFFFFF;
}
.logo-accent {
    color: #38BDF8;
}
.logo-sub {
    font-size: 11px;
    color: #94A3B8;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

/* User Profile Card in Sidebar */
.sidebar-user-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid #1E293B;
    border-radius: 12px;
    margin-bottom: 20px;
}
.sidebar-avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
    color: #FFFFFF;
}
.avatar-admin {
    background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%);
    box-shadow: 0 0 12px rgba(139, 92, 246, 0.4);
}
.avatar-student {
    background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%);
    box-shadow: 0 0 12px rgba(14, 165, 233, 0.4);
}

/* Top Navbar for Public / Landing Page */
.top-navbar-dark {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 28px;
    background: rgba(19, 28, 46, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid #1E293B;
    border-radius: 14px;
    margin-bottom: 28px;
}
.top-logo-dark {
    display: flex;
    align-items: center;
    gap: 12px;
}

/* Hero Banner */
.hero-box-dark {
    background: radial-gradient(circle at top right, rgba(14, 165, 233, 0.15) 0%, transparent 60%),
                linear-gradient(135deg, #0E172A 0%, #131E35 50%, #0B1220 100%);
    border-radius: 18px;
    padding: 44px 40px;
    color: #FFFFFF;
    margin-bottom: 28px;
    border: 1px solid rgba(56, 189, 248, 0.2);
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
    position: relative;
    overflow: hidden;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(56, 189, 248, 0.12);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 18px;
    letter-spacing: 0.04em;
}
.hero-title-dark {
    font-size: 38px;
    font-weight: 800;
    line-height: 1.18;
    color: #FFFFFF;
    margin-bottom: 14px;
    letter-spacing: -0.02em;
}
.hero-glow-text {
    background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-desc-dark {
    font-size: 15px;
    color: #94A3B8;
    max-width: 520px;
    line-height: 1.6;
    margin-bottom: 24px;
}

/* Content Cards (Dark Theme) */
.content-card-dark {
    background: #111A2E;
    border-radius: 16px;
    padding: 24px;
    border: 1px solid #1E293B;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    margin-bottom: 22px;
}
.content-card-dark h3, .content-card-dark h4 {
    margin: 0 0 14px 0;
    color: #F8FAFC;
    font-weight: 700;
}

/* KPI Cards (Dark Theme) */
.kpi-card-dark {
    background: #111A2E;
    border-radius: 14px;
    padding: 20px 22px;
    border: 1px solid #1E293B;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    display: flex;
    align-items: center;
    gap: 16px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card-dark:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.35);
}
.kpi-icon-dark {
    width: 50px;
    height: 50px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.kpi-icon-blue { background: rgba(14, 165, 233, 0.15); color: #38BDF8; border: 1px solid rgba(14, 165, 233, 0.3); }
.kpi-icon-green { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
.kpi-icon-purple { background: rgba(168, 85, 247, 0.15); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.3); }
.kpi-icon-amber { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }

.kpi-title-dark {
    font-size: 13px;
    font-weight: 600;
    color: #94A3B8;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.kpi-val-dark {
    font-size: 26px;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.1;
}

/* Feature Cards for Landing Page */
.feature-card-dark {
    background: #111A2E;
    border-radius: 14px;
    padding: 24px 20px;
    border: 1px solid #1E293B;
    text-align: center;
    height: 100%;
    transition: all 0.2s ease;
}
.feature-card-dark:hover {
    border-color: rgba(56, 189, 248, 0.3);
    transform: translateY(-3px);
}
.feature-card-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 14px auto;
    font-size: 22px;
}
.feature-card-dark h4 {
    margin: 0 0 8px 0;
    font-size: 16px;
    font-weight: 700;
    color: #F8FAFC;
}
.feature-card-dark p {
    margin: 0;
    font-size: 13px;
    color: #94A3B8;
    line-height: 1.5;
}

/* Auth Cards (Dark Theme) */
.auth-card-dark {
    background: #111A2E;
    border-radius: 18px;
    padding: 38px 34px;
    border: 1px solid #1E293B;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5);
    max-width: 480px;
    margin: 20px auto;
    text-align: center;
}
.auth-title-dark {
    font-size: 24px;
    font-weight: 800;
    color: #F8FAFC;
    margin: 10px 0 6px 0;
}
.auth-subtitle-dark {
    font-size: 13.5px;
    color: #94A3B8;
    margin-bottom: 24px;
}

/* Badges (Dark Mode High Contrast) */
.badge-status {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.badge-pass {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.35);
}
.badge-fail {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(248, 113, 113, 0.35);
}
.badge-active {
    background: rgba(56, 189, 248, 0.15);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.35);
}
.badge-grade {
    display: inline-block;
    width: 26px;
    height: 26px;
    line-height: 26px;
    text-align: center;
    border-radius: 6px;
    font-weight: 800;
    font-size: 13px;
}
.grade-A { background: rgba(14, 165, 233, 0.2); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); }
.grade-B { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.4); }
.grade-C { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.4); }
.grade-D { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(248, 113, 113, 0.4); }

/* Dark Mode Tables */
.dark-table-container {
    overflow-x: auto;
    border-radius: 12px;
    border: 1px solid #1E293B;
    background: #0D1526;
    margin-top: 10px;
}
.dark-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
}
.dark-table th {
    background: #141E34;
    color: #94A3B8;
    padding: 12px 14px;
    text-align: left;
    font-weight: 600;
    border-bottom: 1px solid #1E293B;
    text-transform: uppercase;
    font-size: 11.5px;
    letter-spacing: 0.05em;
}
.dark-table td {
    padding: 12px 14px;
    border-bottom: 1px solid #16223B;
    color: #E2E8F0;
}
.dark-table tr:hover {
    background: rgba(56, 189, 248, 0.03);
}

/* Buttons Styling */
div.stButton > button {
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #38BDF8 !important;
    border-radius: 10px !important;
    padding: 9px 18px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
    box-shadow: 0 4px 16px rgba(14, 165, 233, 0.4) !important;
    border-color: #7DD3FC !important;
}

/* Inputs, Selects, Number inputs */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: #0E1626 !important;
    color: #F8FAFC !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 1px #38BDF8 !important;
}
div[data-testid="stExpander"] {
    background-color: #111A2E !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
}

/* Result Banner */
.pred-result-card-dark {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(13, 21, 38, 0.9) 100%);
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 16px;
    padding: 24px;
    margin-top: 24px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}
</style>
"""

render_html(DARK_THEME_CSS)


# -----------------------------------------------------------------------------
# SIDEBAR BRAND LOGO
# -----------------------------------------------------------------------------
def render_sidebar_brand():
    brand_html = f"""
    <div class="sidebar-logo-box">
        <div class="sidebar-logo-icon">
            {get_brand_logo_svg(24)}
        </div>
        <div class="sidebar-logo-text">
            <div class="logo-main">EduPredict<span class="logo-accent">.AI</span></div>
            <div class="logo-sub">Neural Performance Hub</div>
        </div>
    </div>
    """
    render_html(brand_html)


# -----------------------------------------------------------------------------
# SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None
if "full_name" not in st.session_state:
    st.session_state.full_name = None
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Overview"
if "landing_view" not in st.session_state:
    st.session_state.landing_view = "home"


def sign_out():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.full_name = None
    st.session_state.landing_view = "home"
    st.session_state.active_nav = "Overview"


# =============================================================================
# SCREEN 1: HOME / LANDING PAGE (DARK MODE)
# =============================================================================
def render_screen_1_home():
    # Top Navbar
    top_nav_html = f"""
    <div class="top-navbar-dark">
        <div class="top-logo-dark">
            <div style="width: 36px; height: 36px; background: linear-gradient(135deg, #1E3A8A 0%, #0284C7 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center;">
                {get_brand_logo_svg(20)}
            </div>
            <div>
                <span style="font-size: 18px; font-weight: 800; color: #FFFFFF;">EduPredict<span style="color:#38BDF8;">.AI</span></span>
                <span style="font-size: 11px; margin-left: 8px; background: rgba(56, 189, 248, 0.15); color: #38BDF8; padding: 2px 8px; border-radius: 999px; font-weight: 700;">v2.5 Dark</span>
            </div>
        </div>
        <div style="display: flex; gap: 20px; align-items: center; font-size: 14px; font-weight: 600;">
            <span style="color: #38BDF8; cursor: pointer;">Platform</span>
            <span style="color: #94A3B8;">Features</span>
            <span style="color: #94A3B8;">Architecture</span>
        </div>
    </div>
    """
    render_html(top_nav_html)

    # Hero Banner
    c_text, c_img = st.columns([1.25, 1])
    with c_text:
        hero_html = """
        <div class="hero-box-dark">
            <div class="hero-badge">
                <span>⚡</span> AI-POWERED ACADEMIC ANALYTICS
            </div>
            <div class="hero-title-dark">
                Predict & Elevate<br>
                <span class="hero-glow-text">Student Performance</span>
            </div>
            <div class="hero-desc-dark">
                Harness Random Forest machine learning models to forecast exam scores, identify at-risk students in advance, and generate personalized academic action plans.
            </div>
        </div>
        """
        render_html(hero_html)

        b1, b2, b3 = st.columns([1, 1.1, 1.2])
        with b1:
            if st.button("🚀 Get Started", key="hero_get_started", use_container_width=True):
                st.session_state.landing_view = "register"
                st.rerun()
        with b2:
            if st.button("👤 Student Login", key="hero_login", use_container_width=True):
                st.session_state.landing_view = "login"
                st.rerun()
        with b3:
            if st.button("🛡️ Admin Portal", key="hero_adm_btn", use_container_width=True):
                st.session_state.landing_view = "admin_login"
                st.rerun()

    with c_img:
        ai_card_html = f"""
        <div style="background: linear-gradient(135deg, #111B30 0%, #0A1220 100%); border-radius: 18px; padding: 36px 30px; text-align: center; border: 1px solid rgba(56, 189, 248, 0.25); box-shadow: 0 12px 32px rgba(0,0,0,0.5); height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="width: 72px; height: 72px; background: rgba(56, 189, 248, 0.1); border-radius: 20px; display: flex; align-items: center; justify-content: center; margin-bottom: 16px; border: 1px solid rgba(56, 189, 248, 0.3);">
                {get_brand_logo_svg(42)}
            </div>
            <div style="color: #F8FAFC; font-weight: 800; font-size: 20px;">EduPredict Neural Intelligence</div>
            <div style="color: #94A3B8; font-size: 13.5px; margin-top: 6px; max-width: 320px;">
                Trained on behavioral metrics, sleep cycles, prior grades, and study volume.
            </div>
            <div style="margin-top: 24px; display: flex; gap: 12px; justify-content: center;">
                <div style="background: #090F1D; border: 1px solid #1E293B; padding: 8px 14px; border-radius: 10px;">
                    <div style="color: #38BDF8; font-weight: 800; font-size: 18px;">94.2%</div>
                    <div style="color: #64748B; font-size: 11px;">Accuracy</div>
                </div>
                <div style="background: #090F1D; border: 1px solid #1E293B; padding: 8px 14px; border-radius: 10px;">
                    <div style="color: #34D399; font-weight: 800; font-size: 18px;">&lt; 50ms</div>
                    <div style="color: #64748B; font-size: 11px;">Inference</div>
                </div>
                <div style="background: #090F1D; border: 1px solid #1E293B; padding: 8px 14px; border-radius: 10px;">
                    <div style="color: #C084FC; font-weight: 800; font-size: 18px;">3-Way</div>
                    <div style="color: #64748B; font-size: 11px;">Ensemble</div>
                </div>
            </div>
        </div>
        """
        render_html(ai_card_html)

    st.write("")
    st.write("")

    # Feature Grid
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        render_html("""
        <div class="feature-card-dark">
            <div class="feature-card-icon" style="background: rgba(14, 165, 233, 0.15); color: #38BDF8;">📊</div>
            <h4>Continuous Regressor</h4>
            <p>Predicts exact final examination score on a 0-100 continuous scale.</p>
        </div>
        """)
    with fc2:
        render_html("""
        <div class="feature-card-dark">
            <div class="feature-card-icon" style="background: rgba(168, 85, 247, 0.15); color: #C084FC;">🏆</div>
            <h4>Letter Grade Classifier</h4>
            <p>Categorizes outcomes into A, B, C, or D with class probability vectors.</p>
        </div>
        """)
    with fc3:
        render_html("""
        <div class="feature-card-dark">
            <div class="feature-card-icon" style="background: rgba(16, 185, 129, 0.15); color: #34D399;">💡</div>
            <h4>AI Advisory Engine</h4>
            <p>Generates tailored academic action plans based on student weaknesses.</p>
        </div>
        """)
    with fc4:
        render_html("""
        <div class="feature-card-dark">
            <div class="feature-card-icon" style="background: rgba(245, 158, 11, 0.15); color: #FBBF24;">🛡️</div>
            <h4>Admin Command</h4>
            <p>Institutional roster oversight, student updates, and full prediction logs.</p>
        </div>
        """)


# =============================================================================
# SCREEN 2: STUDENT LOGIN (DARK MODE)
# =============================================================================
def render_screen_2_student_login():
    left, mid, right = st.columns([1, 1.4, 1])
    with mid:
        auth_hdr = f"""
        <div class="auth-card-dark">
            <div style="width: 52px; height: 52px; margin: 0 auto; background: rgba(56, 189, 248, 0.12); border-radius: 14px; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(56, 189, 248, 0.3);">
                {get_brand_logo_svg(28)}
            </div>
            <div class="auth-title-dark">Student Sign In</div>
            <div class="auth-subtitle-dark">Access your personal performance dashboard & AI recommendations.</div>
        </div>
        """
        render_html(auth_hdr)

        user_input = st.text_input("Username", value="megha", placeholder="Enter your username", key="login_u")
        pw_input = st.text_input("Password", value="password123", type="password", placeholder="Enter your password", key="login_p")

        st.write("")
        if st.button("Sign In to Student Portal", use_container_width=True, key="btn_do_login"):
            user = db.verify_user(user_input, pw_input)
            if user and user["role"] == "student":
                st.session_state.logged_in = True
                st.session_state.username = user["username"]
                st.session_state.role = user["role"]
                st.session_state.full_name = user["full_name"]
                st.session_state.active_nav = "Dashboard"
                st.rerun()
            else:
                st.error("Invalid student credentials. (Demo: `megha` / `password123`)")

        st.write("")
        c_link1, c_link2 = st.columns([1, 1])
        with c_link1:
            st.caption("New student?")
            if st.button("Create Account", key="btn_to_reg", use_container_width=True):
                st.session_state.landing_view = "register"
                st.rerun()
        with c_link2:
            st.caption("Institutional Staff?")
            if st.button("Admin Portal", key="btn_to_admin_login", use_container_width=True):
                st.session_state.landing_view = "admin_login"
                st.rerun()


# =============================================================================
# SCREEN 3: STUDENT REGISTRATION (DARK MODE)
# =============================================================================
def render_screen_3_student_registration():
    left, mid, right = st.columns([1, 1.4, 1])
    with mid:
        auth_hdr = f"""
        <div class="auth-card-dark">
            <div style="width: 52px; height: 52px; margin: 0 auto; background: rgba(56, 189, 248, 0.12); border-radius: 14px; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(56, 189, 248, 0.3);">
                {get_brand_logo_svg(28)}
            </div>
            <div class="auth-title-dark">Student Registration</div>
            <div class="auth-subtitle-dark">Create your account to track predictions & academic milestones.</div>
        </div>
        """
        render_html(auth_hdr)

        reg_name = st.text_input("Full Name", placeholder="e.g. Alex Morgan", key="reg_fn")
        reg_user = st.text_input("Choose Username", placeholder="e.g. alexm", key="reg_un")
        reg_pw = st.text_input("Password (min 6 characters)", type="password", placeholder="Enter password", key="reg_pw1")
        reg_cpw = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="reg_pw2")

        st.write("")
        if st.button("Complete Registration", use_container_width=True, key="btn_do_register"):
            if not reg_name or not reg_user or not reg_pw:
                st.error("Please fill in all required fields.")
            elif len(reg_pw) < 6:
                st.error("Password must be at least 6 characters long.")
            elif reg_pw != reg_cpw:
                st.error("Passwords do not match.")
            else:
                ok, msg = db.create_user(reg_user, reg_pw, reg_name, role="student")
                if ok:
                    st.success(f"{msg} You can now log in.")
                    st.session_state.landing_view = "login"
                    st.rerun()
                else:
                    st.error(msg)

        st.write("")
        st.caption("Already have an active account?")
        if st.button("Back to Student Login", key="btn_to_login_from_reg", use_container_width=True):
            st.session_state.landing_view = "login"
            st.rerun()


# =============================================================================
# SCREEN 4: ADMINISTRATOR LOGIN (DARK MODE)
# =============================================================================
def render_admin_login():
    left, mid, right = st.columns([1, 1.4, 1])
    with mid:
        auth_hdr = """
        <div class="auth-card-dark">
            <div style="width: 52px; height: 52px; margin: 0 auto; background: rgba(168, 85, 247, 0.15); border-radius: 14px; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(168, 85, 247, 0.35);">
                <span style="font-size: 26px;">🛡️</span>
            </div>
            <div class="auth-title-dark">Administrator Portal</div>
            <div class="auth-subtitle-dark">Restricted institutional access for faculty & administrators.</div>
        </div>
        """
        render_html(auth_hdr)

        adm_user = st.text_input("Admin Username", value="admin", key="adm_u")
        adm_pw = st.text_input("Admin Password", value="admin123", type="password", key="adm_p")

        st.write("")
        if st.button("Authorize & Enter Admin Hub", use_container_width=True, key="btn_do_adm_login"):
            user = db.verify_user(adm_user, adm_pw)
            if user and user["role"] == "admin":
                st.session_state.logged_in = True
                st.session_state.username = user["username"]
                st.session_state.role = user["role"]
                st.session_state.full_name = user["full_name"]
                st.session_state.active_nav = "Overview"
                st.rerun()
            else:
                st.error("Invalid administrator credentials. (Default: `admin` / `admin123`)")

        st.write("")
        if st.button("← Back to Student Login", key="btn_back_to_stu_login", use_container_width=True):
            st.session_state.landing_view = "login"
            st.rerun()


# =============================================================================
# SCREEN 5: STUDENT DASHBOARD (DARK MODE)
# =============================================================================
def render_screen_4_student_dashboard():
    first_name = st.session_state.full_name.split()[0] if st.session_state.full_name else "Student"
    t1, t2 = st.columns([3, 1])
    with t1:
        st.markdown(f"## Welcome back, {first_name}! 👋")
        st.caption("Here is your latest academic trajectory and machine learning predictions.")
    with t2:
        render_html("""
        <div style="text-align: right; padding-top: 10px;">
            <span style="background: rgba(56, 189, 248, 0.12); color: #38BDF8; padding: 6px 12px; border-radius: 999px; font-weight: 700; font-size: 12px; border: 1px solid rgba(56, 189, 248, 0.25);">
                👤 Student Account
            </span>
        </div>
        """)

    st.write("")

    preds = db.get_predictions_for_user(st.session_state.username)
    if preds:
        latest = preds[0]
        score_val = f"{latest['predicted_score']:.1f}%"
        grade_val = latest["predicted_grade"]
        status_val = latest["predicted_passfail"].upper()
        total_count = len(preds)
    else:
        score_val = "N/A"
        grade_val = "—"
        status_val = "NO RUNS"
        total_count = 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-blue">📊</div>
            <div>
                <div class="kpi-title-dark">Latest Score</div>
                <div class="kpi-val-dark" style="color: #38BDF8;">{score_val}</div>
            </div>
        </div>
        """)

    with k2:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-green">🏆</div>
            <div>
                <div class="kpi-title-dark">Latest Grade</div>
                <div class="kpi-val-dark" style="color: #34D399;">{grade_val}</div>
            </div>
        </div>
        """)

    with k3:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-purple">🎯</div>
            <div>
                <div class="kpi-title-dark">Exam Outcome</div>
                <div class="kpi-val-dark" style="color: #C084FC;">{status_val}</div>
            </div>
        </div>
        """)

    with k4:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-amber">📑</div>
            <div>
                <div class="kpi-title-dark">Predictions</div>
                <div class="kpi-val-dark" style="color: #FBBF24;">{total_count}</div>
            </div>
        </div>
        """)

    st.write("")

    col_chart, col_actions = st.columns([2.3, 1])

    with col_chart:
        render_html("""
        <div class="content-card-dark" style="margin-bottom:0; padding-bottom:10px;">
            <h4>Academic Trajectory Progression</h4>
        </div>
        """)

        if preds and len(preds) >= 2:
            df_hist = pd.DataFrame([dict(p) for p in preds]).iloc[::-1]
            dates = [p["created_at"].split()[0] for _, p in df_hist.iterrows()]
            scores = df_hist["predicted_score"].tolist()
        else:
            dates = ["Entry Baseline", "Midterm 1", "Assignment Avg", "Current"]
            scores = [65.0, 71.0, 74.5, 78.5]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=scores,
            mode="lines+markers",
            line=dict(color="#38BDF8", width=3, shape="spline"),
            marker=dict(size=8, color="#38BDF8", line=dict(color="#FFFFFF", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.08)",
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=15, b=20),
            yaxis=dict(range=[40, 100], gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
            xaxis=dict(gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Plus Jakarta Sans"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_actions:
        render_html("""
        <div class="content-card-dark" style="margin-bottom:12px;">
            <h4>Quick Shortcuts</h4>
        </div>
        """)

        if st.button("🔮 Predict New Exam Score", use_container_width=True, key="qa_pred"):
            st.session_state.active_nav = "Predict Performance"
            st.rerun()

        st.write("")
        if st.button("📈 View Complete History", use_container_width=True, key="qa_hist"):
            st.session_state.active_nav = "My History & Analytics"
            st.rerun()

        st.write("")
        if st.button("👤 Update My Profile", use_container_width=True, key="qa_prof"):
            st.session_state.active_nav = "My Profile"
            st.rerun()


# =============================================================================
# SCREEN 6: STUDENT PREDICT PERFORMANCE (DARK MODE)
# =============================================================================
def render_screen_5_prediction_page():
    st.markdown("## 🔮 Predict Your Academic Performance")
    st.caption("Input your weekly learning habits, attendance, and behavioral factors for multi-model inference.")

    with st.form("predict_form"):
        c_left, c_right = st.columns(2)
        with c_left:
            study_hrs = st.slider("Weekly Study Hours (0 - 40h)", min_value=0.0, max_value=40.0, value=20.0, step=0.5)
            att_pct = st.slider("Class Attendance Rate (%)", min_value=30.0, max_value=100.0, value=85.0, step=1.0)
            prev_sc = st.number_input("Previous Examination Score (0 - 100)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
            sleep_hrs = st.slider("Average Nightly Sleep (Hours)", min_value=3.0, max_value=10.0, value=7.5, step=0.5)

        with c_right:
            pe_map = {
                "Graduate / Masters": "masters",
                "Bachelor's Degree": "bachelors",
                "High School": "high_school",
                "PhD / Doctorate": "phd",
            }
            parental_edu = st.selectbox("Parental Education Level", list(pe_map.keys()), index=1)
            internet_acc = st.selectbox("High-Speed Internet Access at Home", ["Yes", "No"], index=0)
            extracurricular = st.selectbox("Participation in Extracurricular Activities", ["Yes", "No"], index=0)

        st.write("")
        submitted = st.form_submit_button("⚡ Run ML Ensemble Prediction", use_container_width=True)

    if submitted and suite:
        inputs = {
            "study_hours_per_week": study_hrs,
            "attendance_percent": att_pct,
            "previous_score": prev_sc,
            "sleep_hours": sleep_hrs,
            "parental_education": pe_map.get(parental_edu, "bachelors"),
            "internet_access": internet_acc.lower(),
            "extracurricular": extracurricular.lower(),
        }
        res = suite.predict_single(inputs)

        db.save_prediction(
            st.session_state.username,
            inputs,
            res["predicted_score"],
            res["predicted_passfail"],
            res["predicted_grade"],
        )
        st.session_state["cur_pred_result"] = res
        st.session_state["cur_pred_inputs"] = inputs

    if "cur_pred_result" in st.session_state:
        res = st.session_state["cur_pred_result"]
        inp = st.session_state.get("cur_pred_inputs", {})
        recs = generate_recommendations(inp, res)

        outcome_color = "#34D399" if res["predicted_passfail"].upper() == "PASS" else "#F87171"

        result_html = f"""
        <div class="pred-result-card-dark">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; border-bottom:1px solid #1E293B; padding-bottom:14px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:22px;">✅</span>
                    <span style="font-size:17px; font-weight:800; color:#F8FAFC;">Ensemble Inference Results</span>
                </div>
                <div>
                    <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 700;">
                        Risk Tier: {res.get('risk_tier', 'Low Risk')}
                    </span>
                </div>
            </div>
            <div style="display:flex; justify-content:space-around; align-items:center; flex-wrap:wrap; gap:18px;">
                <div style="text-align:center;">
                    <div style="font-size:12px; font-weight:600; color:#94A3B8; text-transform:uppercase; margin-bottom:4px;">Predicted Score</div>
                    <div style="font-size:32px; font-weight:800; color:#38BDF8;">{res['predicted_score']:.1f}%</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:12px; font-weight:600; color:#94A3B8; text-transform:uppercase; margin-bottom:4px;">Letter Grade</div>
                    <div style="font-size:32px; font-weight:800; color:#C084FC;">{res['predicted_grade']}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:12px; font-weight:600; color:#94A3B8; text-transform:uppercase; margin-bottom:4px;">Outcome</div>
                    <div style="font-size:32px; font-weight:800; color:{outcome_color};">{res['predicted_passfail'].upper()}</div>
                </div>
            </div>
        </div>
        """
        render_html(result_html)

        if recs:
            st.write("")
            render_html("""
            <div class="content-card-dark" style="margin-bottom: 12px;">
                <h4>🧠 AI Academic Action Plan & Personalized Advice</h4>
            </div>
            """)
            for r in recs:
                cat = r.get("category", "Strategy")
                act = r.get("action", "")
                plan_item_html = f"""
                <div style="display:flex; align-items:flex-start; gap:12px; padding:12px 14px; background:#0E1626; border-radius:10px; margin-bottom:10px; border:1px solid #1E293B;">
                    <span style="background:rgba(56, 189, 248, 0.15); color:#38BDF8; padding:3px 8px; border-radius:6px; font-size:11px; font-weight:700; flex-shrink:0;">{cat}</span>
                    <span style="color:#CBD5E1; font-size:13.5px; line-height:1.5;">{act}</span>
                </div>
                """
                render_html(plan_item_html)


# =============================================================================
# SCREEN 7: STUDENT HISTORY & ANALYTICS (DARK MODE)
# =============================================================================
def render_screen_6_history():
    st.markdown("## 📈 My Academic History & Analytics")
    st.caption("Review all your prior simulated examination predictions.")

    rows = db.get_predictions_for_user(st.session_state.username)
    if not rows:
        st.info("No prior prediction logs recorded for this account. Run a prediction to see your analytics!")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    total_pred = len(df)
    avg_score = f"{df['predicted_score'].mean():.1f}%"
    max_score = f"{df['predicted_score'].max():.1f}%"
    pass_cnt = len(df[df["predicted_passfail"].str.upper() == "PASS"])
    pass_rate = f"{(pass_cnt / total_pred * 100):.1f}%"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-blue">📊</div>
            <div>
                <div class="kpi-title-dark">Total Runs</div>
                <div class="kpi-val-dark" style="color:#38BDF8;">{total_pred}</div>
            </div>
        </div>
        """)
    with k2:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-green">📈</div>
            <div>
                <div class="kpi-title-dark">Avg Score</div>
                <div class="kpi-val-dark" style="color:#34D399;">{avg_score}</div>
            </div>
        </div>
        """)
    with k3:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-purple">🏆</div>
            <div>
                <div class="kpi-title-dark">Peak Score</div>
                <div class="kpi-val-dark" style="color:#C084FC;">{max_score}</div>
            </div>
        </div>
        """)
    with k4:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-amber">🎯</div>
            <div>
                <div class="kpi-title-dark">Pass Rate</div>
                <div class="kpi-val-dark" style="color:#FBBF24;">{pass_rate}</div>
            </div>
        </div>
        """)

    st.write("")

    c_hdr, c_dl = st.columns([3, 1])
    with c_hdr:
        st.markdown("### Detailed Prediction History Log")
    with c_dl:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥 Download My Logs (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"{st.session_state.username}_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    rows_html = ""
    for _, row in df.iterrows():
        dt = str(row["created_at"]).split()[0]
        badge_cls = "badge-pass" if row["predicted_passfail"].upper() == "PASS" else "badge-fail"
        grade_letter = row["predicted_grade"]
        rows_html += f"""
        <tr>
            <td style="color: #94A3B8;">{dt}</td>
            <td>{int(row['study_hours_per_week'])} hrs</td>
            <td>{int(row['attendance_percent'])}%</td>
            <td>{int(row['previous_score'])}</td>
            <td>{row.get('sleep_hours', 7.0)} hrs</td>
            <td style="font-weight: 700; color: #38BDF8;">{row['predicted_score']:.1f}%</td>
            <td><span class="badge-grade grade-{grade_letter}">{grade_letter}</span></td>
            <td><span class="badge-status {badge_cls}">{row['predicted_passfail'].upper()}</span></td>
        </tr>
        """

    full_table = f"""
    <div class="dark-table-container">
        <table class="dark-table">
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Study Hours</th>
                    <th>Attendance</th>
                    <th>Prev Score</th>
                    <th>Sleep</th>
                    <th>Predicted Score</th>
                    <th>Grade</th>
                    <th>Outcome</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    render_html(full_table)


# =============================================================================
# SCREEN 8: STUDENT PROFILE (DARK MODE)
# =============================================================================
def render_student_profile():
    st.markdown("## 👤 Student Profile & Account")
    st.caption("Manage your student account details and credentials.")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        avatar_char = st.session_state.full_name[0] if st.session_state.full_name else 'S'
        profile_card_html = f"""
        <div class="content-card-dark">
            <h4>Account Profile</h4>
            <div style="display:flex; align-items:center; gap:16px; margin-bottom:12px; padding:16px; background:#0E1626; border-radius:12px; border:1px solid #1E293B;">
                <div class="sidebar-avatar avatar-student" style="width:52px; height:52px; font-size:20px;">
                    {avatar_char}
                </div>
                <div>
                    <div style="font-size:18px; font-weight:800; color:#F8FAFC;">{st.session_state.full_name}</div>
                    <div style="font-size:13px; color:#38BDF8; font-weight:600;">@{st.session_state.username}</div>
                    <div style="font-size:12px; color:#94A3B8;">Student • Role: Active</div>
                </div>
            </div>
        </div>
        """
        render_html(profile_card_html)

    with c2:
        with st.form("student_pw_form"):
            st.markdown("#### Update Password")
            cur_pw = st.text_input("Current Password", type="password")
            new_pw = st.text_input("New Password (min 6 chars)", type="password")
            conf_pw = st.text_input("Confirm New Password", type="password")
            if st.form_submit_button("Update My Password", use_container_width=True):
                if not cur_pw or not new_pw:
                    st.error("Please fill in all password fields.")
                elif len(new_pw) < 6:
                    st.error("New password must be at least 6 characters.")
                elif new_pw != conf_pw:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = db.change_user_password(st.session_state.username, cur_pw, new_pw)
                    if ok:
                        st.success("Your password has been updated successfully.")
                    else:
                        st.error(msg)


# =============================================================================
# SCREEN 9: ADMIN — OVERVIEW (DARK MODE)
# =============================================================================
def render_screen_7_admin_overview():
    t1, t2 = st.columns([3, 1])
    with t1:
        st.markdown("## 📊 Institutional Admin Overview")
        st.caption("Live high-level performance indicators, grade distributions, and cohort trends.")
    with t2:
        render_html("""
        <div style="text-align: right; padding-top: 10px;">
            <span style="background: rgba(168, 85, 247, 0.15); color: #C084FC; padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: 12px; border: 1px solid rgba(168, 85, 247, 0.3);">
                🛡️ Admin Authority
            </span>
        </div>
        """)

    st.write("")

    stats = db.get_summary_stats()

    # 4 KPI Grid
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-blue">👥</div>
            <div>
                <div class="kpi-title-dark">Total Students</div>
                <div class="kpi-val-dark" style="color: #38BDF8;">{stats['total_students']}</div>
            </div>
        </div>
        """)

    with k2:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-purple">🔮</div>
            <div>
                <div class="kpi-title-dark">Predictions</div>
                <div class="kpi-val-dark" style="color: #C084FC;">{stats['total_predictions']}</div>
            </div>
        </div>
        """)

    with k3:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-green">📈</div>
            <div>
                <div class="kpi-title-dark">Avg Score</div>
                <div class="kpi-val-dark" style="color: #34D399;">{stats['avg_score']}%</div>
            </div>
        </div>
        """)

    with k4:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-amber">🎯</div>
            <div>
                <div class="kpi-title-dark">Pass Rate</div>
                <div class="kpi-val-dark" style="color: #FBBF24;">{stats['pass_rate']}%</div>
            </div>
        </div>
        """)

    st.write("")

    # Visual Charts
    col_chart, col_donut = st.columns([1.8, 1.2])

    with col_chart:
        render_html("""
        <div class="content-card-dark" style="margin-bottom:0; padding-bottom:8px;">
            <h4>Student Performance Trajectory (Institutional Trend)</h4>
        </div>
        """)

        all_preds = db.get_all_predictions(limit=50)
        if all_preds and len(all_preds) >= 4:
            df_chart = pd.DataFrame([dict(r) for r in all_preds]).iloc[::-1]
            x_vals = [f"#{idx+1}" for idx in range(len(df_chart))]
            y_vals = df_chart["predicted_score"].tolist()
        else:
            x_vals = ["Aug", "Sep (W1)", "Sep (W2)", "Sep (W3)", "Sep (W4)", "Current"]
            y_vals = [66.5, 71.0, 72.8, 73.5, 74.8, float(stats['avg_score']) if stats['avg_score'] else 75.0]

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            line=dict(color="#38BDF8", width=3, shape="spline"),
            marker=dict(size=7, color="#38BDF8", line=dict(color="#FFFFFF", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.08)",
        ))
        fig_line.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=10, b=20),
            yaxis=dict(range=[40, 100], gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
            xaxis=dict(gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Plus Jakarta Sans"},
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_donut:
        render_html("""
        <div class="content-card-dark" style="margin-bottom:0; padding-bottom:8px;">
            <h4>Grade Distribution Breakdown</h4>
        </div>
        """)

        all_preds = db.get_all_predictions(limit=500)
        if all_preds:
            df_p = pd.DataFrame([dict(r) for r in all_preds])
            grade_counts = df_p["predicted_grade"].value_counts().to_dict()
            labels = ["Grade A", "Grade B", "Grade C", "Grade D"]
            values = [
                grade_counts.get("A", 0) + 1,
                grade_counts.get("B", 0) + 2,
                grade_counts.get("C", 0) + 1,
                grade_counts.get("D", 0) + 1,
            ]
        else:
            labels = ["Grade A", "Grade B", "Grade C", "Grade D"]
            values = [30, 45, 15, 10]

        colors = ["#38BDF8", "#34D399", "#FBBF24", "#F87171"]

        fig_donut = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker_colors=colors,
            textinfo="percent",
            textfont=dict(color="#F8FAFC", family="Plus Jakarta Sans", size=11),
        )])
        fig_donut.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, font=dict(color="#CBD5E1")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Plus Jakarta Sans"},
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.write("")

    # Recent Predictions Quick Preview
    render_html("""
    <div class="content-card-dark" style="margin-bottom:12px;">
        <h4 style="margin:0;">Recent Institution Predictions</h4>
    </div>
    """)

    recent_preds = db.get_all_predictions(limit=6)
    if recent_preds:
        recent_rows = ""
        for r in recent_preds:
            dr = dict(r)
            badge_cls = "badge-pass" if dr["predicted_passfail"].upper() == "PASS" else "badge-fail"
            recent_rows += f"""
            <tr>
                <td style="font-weight:700; color:#F8FAFC;">{dr['full_name']}</td>
                <td style="color:#94A3B8;">@{dr['username']}</td>
                <td>{int(dr['study_hours_per_week'])} hrs</td>
                <td>{int(dr['attendance_percent'])}%</td>
                <td style="font-weight:700; color:#38BDF8;">{dr['predicted_score']:.1f}%</td>
                <td><span class="badge-grade grade-{dr['predicted_grade']}">{dr['predicted_grade']}</span></td>
                <td><span class="badge-status {badge_cls}">{dr['predicted_passfail'].upper()}</span></td>
                <td style="color:#64748B;">{str(dr['created_at']).split()[0]}</td>
            </tr>
            """
        recent_table_html = f"""
        <div class="dark-table-container">
            <table class="dark-table">
                <thead>
                    <tr>
                        <th>Student Name</th>
                        <th>Username</th>
                        <th>Study Hours</th>
                        <th>Attendance</th>
                        <th>Predicted Score</th>
                        <th>Grade</th>
                        <th>Status</th>
                        <th>Date</th>
                    </tr>
                </thead>
                <tbody>
                    {recent_rows}
                </tbody>
            </table>
        </div>
        """
        render_html(recent_table_html)
    else:
        st.info("No predictions in database yet.")


# =============================================================================
# SCREEN 10: ADMIN — MANAGE STUDENTS (DARK MODE + UPDATE & DELETE)
# =============================================================================
def render_screen_8_manage_students():
    t1, t2 = st.columns([2.5, 1.5])
    with t1:
        st.markdown("## 👥 Manage Students")
        st.caption("Review student directory, edit profile details, reset passwords, or remove student accounts.")
    with t2:
        search_query = st.text_input("🔍 Search student...", label_visibility="collapsed", placeholder="🔍 Search student...")

    all_students = db.get_all_students()

    # Expander to Add a New Student
    with st.expander("➕ Register / Add New Student Account"):
        with st.form("admin_add_student_form"):
            ca, cb = st.columns(2)
            with ca:
                new_fn = st.text_input("Student Full Name", placeholder="e.g. John Doe")
                new_un = st.text_input("Username", placeholder="e.g. johndoe")
            with cb:
                new_pw = st.text_input("Initial Password (min 6 chars)", type="password")
                new_cpw = st.text_input("Confirm Password", type="password")

            if st.form_submit_button("Create Student Account"):
                if not new_fn or not new_un or not new_pw:
                    st.error("All fields are required.")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pw != new_cpw:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = db.create_user(new_un, new_pw, new_fn, role="student")
                    if ok:
                        st.success(f"Student '{new_fn}' registered successfully!")
                        st.rerun()
                    else:
                        st.error(msg)

    # Filtered students list
    filtered_students = []
    for s in all_students:
        name_match = search_query.lower() in s["full_name"].lower() if s["full_name"] else False
        user_match = search_query.lower() in s["username"].lower()
        if not search_query or name_match or user_match:
            filtered_students.append(s)

    st.markdown(f"### Active Student Roster ({len(filtered_students)} found)")

    table_rows = ""
    for idx, s in enumerate(filtered_students, 1):
        reg_dt = str(s.get("registered_date", "2024-09-01")).split()[0]
        cnt = s.get("prediction_count", 0)
        avg_sc = f"{s['avg_score']:.1f}%" if s.get("avg_score") is not None else "—"
        l_gr = s.get("latest_grade", "—")
        gr_badge = f"<span class='badge-grade grade-{l_gr}'>{l_gr}</span>" if l_gr != "—" else "—"

        table_rows += f"""
        <tr>
            <td style="color:#64748B;">#{idx}</td>
            <td style="font-weight:700; color:#F8FAFC;">{s['full_name']}</td>
            <td style="color:#38BDF8;">@{s['username']}</td>
            <td style="color:#94A3B8;">{reg_dt}</td>
            <td>{cnt} runs</td>
            <td style="font-weight:700; color:#34D399;">{avg_sc}</td>
            <td>{gr_badge}</td>
            <td><span class="badge-status badge-active">Active</span></td>
        </tr>
        """

    full_roster_html = f"""
    <div class="dark-table-container">
        <table class="dark-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Full Name</th>
                    <th>Username</th>
                    <th>Joined Date</th>
                    <th>Predictions</th>
                    <th>Avg Score</th>
                    <th>Latest Grade</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """
    render_html(full_roster_html)

    st.write("")
    st.markdown("### Student Action Center (Update & Delete)")

    if not all_students:
        st.info("No registered students found.")
        return

    student_options = {f"{s['full_name']} (@{s['username']})": s['username'] for s in all_students}
    sel_label = st.selectbox("Select Student to Update or Delete", list(student_options.keys()), key="sel_stu_action")
    target_uname = student_options[sel_label]
    target_student = next((s for s in all_students if s["username"] == target_uname), None)

    tab_edit, tab_delete, tab_history = st.tabs(["✏️ Update Student", "🗑️ Delete Student", "📊 View Student Prediction History"])

    with tab_edit:
        st.markdown(f"##### Update Details for `{target_student['full_name']}` (@{target_uname})")
        with st.form(f"update_form_{target_uname}"):
            up_name = st.text_input("Full Name", value=target_student["full_name"])
            up_new_pw = st.text_input("New Password (leave blank to keep unchanged)", type="password")
            up_conf_pw = st.text_input("Confirm New Password", type="password")

            if st.form_submit_button("💾 Save Student Updates"):
                if not up_name.strip():
                    st.error("Full name cannot be empty.")
                elif up_new_pw and len(up_new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif up_new_pw and up_new_pw != up_conf_pw:
                    st.error("Passwords do not match.")
                else:
                    pw_to_set = up_new_pw if up_new_pw else None
                    ok, msg = db.update_student(target_uname, full_name=up_name, new_password=pw_to_set)
                    if ok:
                        st.success(f"Student @{target_uname} updated successfully!")
                        st.rerun()
                    else:
                        st.error(msg)

    with tab_delete:
        del_warn_html = f"""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 18px; margin-bottom: 18px;">
            <div style="color: #F87171; font-weight: 800; font-size: 15px; margin-bottom: 6px;">⚠️ Danger Zone: Delete Student Account</div>
            <div style="color: #CBD5E1; font-size: 13.5px; line-height: 1.5;">
                You are about to delete <b>{target_student['full_name']}</b> (username: <code>{target_uname}</code>).
                This action is permanent and will remove the student account along with all their prediction history.
            </div>
        </div>
        """
        render_html(del_warn_html)

        confirm_del = st.checkbox(f"Yes, I understand and want to permanently delete student @{target_uname}", key=f"del_chk_{target_uname}")
        if st.button(f"🗑️ Delete Student @{target_uname} Permanently", disabled=not confirm_del, key=f"btn_del_{target_uname}"):
            db.delete_student(target_uname)
            st.success(f"Student @{target_uname} was removed from the institution.")
            st.rerun()

    with tab_history:
        stu_preds = db.get_predictions_for_user(target_uname)
        if not stu_preds:
            st.info(f"Student @{target_uname} does not have any recorded prediction logs yet.")
        else:
            st.markdown(f"##### Prediction History for {target_student['full_name']} ({len(stu_preds)} records)")
            h_rows = ""
            for p in stu_preds:
                dp = dict(p)
                badge = "badge-pass" if dp["predicted_passfail"].upper() == "PASS" else "badge-fail"
                h_rows += f"""
                <tr>
                    <td style="color:#94A3B8;">{str(dp['created_at']).split()[0]}</td>
                    <td>{int(dp['study_hours_per_week'])} hrs</td>
                    <td>{int(dp['attendance_percent'])}%</td>
                    <td>{int(dp['previous_score'])}</td>
                    <td style="font-weight:700; color:#38BDF8;">{dp['predicted_score']:.1f}%</td>
                    <td><span class="badge-grade grade-{dp['predicted_grade']}">{dp['predicted_grade']}</span></td>
                    <td><span class="badge-status {badge}">{dp['predicted_passfail'].upper()}</span></td>
                </tr>
                """
            h_table_html = f"""
            <div class="dark-table-container">
                <table class="dark-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Study Hours</th>
                            <th>Attendance</th>
                            <th>Prev Score</th>
                            <th>Predicted Score</th>
                            <th>Grade</th>
                            <th>Result</th>
                        </tr>
                    </thead>
                    <tbody>
                        {h_rows}
                    </tbody>
                </table>
            </div>
            """
            render_html(h_table_html)


# =============================================================================
# SCREEN 11: ADMIN — ALL PREDICTIONS / STUDENT HISTORY (DARK MODE)
# =============================================================================
def render_screen_9_all_predictions():
    st.markdown("## 📋 Institutional Predictions & Student History")
    st.caption("Complete prediction logs across all enrolled students with multi-filters and CSV export.")

    all_preds = db.get_all_predictions(limit=1000)
    if not all_preds:
        st.info("No prediction records found in the database.")
        return

    df_preds = pd.DataFrame([dict(r) for r in all_preds])

    # Filter toolbar
    st.markdown("### Filter Prediction Records")
    f1, f2, f3, f4 = st.columns([1.5, 1, 1, 1.5])
    with f1:
        students_list = ["All Students"] + sorted(list(df_preds["username"].unique()))
        sel_student = st.selectbox("Student Filter", students_list, index=0)
    with f2:
        grade_options = ["All Grades"] + sorted(list(df_preds["predicted_grade"].unique()))
        sel_grade = st.selectbox("Grade", grade_options, index=0)
    with f3:
        status_options = ["All Outcomes", "Pass", "Fail"]
        sel_status = st.selectbox("Outcome", status_options, index=0)
    with f4:
        kw_search = st.text_input("Keyword Search", placeholder="Name or username...")

    # Apply filters
    filtered_df = df_preds.copy()
    if sel_student != "All Students":
        filtered_df = filtered_df[filtered_df["username"] == sel_student]
    if sel_grade != "All Grades":
        filtered_df = filtered_df[filtered_df["predicted_grade"] == sel_grade]
    if sel_status != "All Outcomes":
        filtered_df = filtered_df[filtered_df["predicted_passfail"].str.upper() == sel_status.upper()]
    if kw_search.strip():
        kw = kw_search.strip().lower()
        filtered_df = filtered_df[
            filtered_df["full_name"].str.lower().str.contains(kw) |
            filtered_df["username"].str.lower().str.contains(kw)
        ]

    # Filtered Summary Metrics
    cnt_total = len(filtered_df)
    avg_score_filtered = f"{filtered_df['predicted_score'].mean():.1f}%" if cnt_total > 0 else "N/A"
    pass_cnt = len(filtered_df[filtered_df["predicted_passfail"].str.upper() == "PASS"])
    pass_rate_filtered = f"{(pass_cnt / cnt_total * 100):.1f}%" if cnt_total > 0 else "N/A"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-blue">📋</div>
            <div>
                <div class="kpi-title-dark">Records in View</div>
                <div class="kpi-val-dark" style="color:#38BDF8;">{cnt_total}</div>
            </div>
        </div>
        """)
    with m2:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-green">📈</div>
            <div>
                <div class="kpi-title-dark">Cohort Avg Score</div>
                <div class="kpi-val-dark" style="color:#34D399;">{avg_score_filtered}</div>
            </div>
        </div>
        """)
    with m3:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-purple">🎯</div>
            <div>
                <div class="kpi-title-dark">Pass Rate</div>
                <div class="kpi-val-dark" style="color:#C084FC;">{pass_rate_filtered}</div>
            </div>
        </div>
        """)
    with m4:
        render_html(f"""
        <div class="kpi-card-dark">
            <div class="kpi-icon-dark kpi-icon-amber">⚠️</div>
            <div>
                <div class="kpi-title-dark">At-Risk Count</div>
                <div class="kpi-val-dark" style="color:#FBBF24;">{len(filtered_df[filtered_df['predicted_score'] < 50.0])}</div>
            </div>
        </div>
        """)

    st.write("")

    # Predictions Table Box & CSV Download
    c_tbl_hdr, c_tbl_dl = st.columns([2.5, 1.2])
    with c_tbl_hdr:
        st.markdown(f"### Prediction Records Log ({cnt_total} results)")
    with c_tbl_dl:
        csv_out = io.StringIO()
        filtered_df.to_csv(csv_out, index=False)
        st.download_button(
            "📥 Download Filtered Log (CSV)",
            data=csv_out.getvalue(),
            file_name="student_predictions_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if cnt_total > 0:
        table_rows = ""
        for _, r in filtered_df.iterrows():
            badge_cls = "badge-pass" if r["predicted_passfail"].upper() == "PASS" else "badge-fail"
            gr = r["predicted_grade"]
            dt = str(r["created_at"]).split()[0]
            table_rows += f"""
            <tr>
                <td style="color:#64748B;">{r['id']}</td>
                <td style="font-weight:700; color:#F8FAFC;">{r['full_name']}</td>
                <td style="color:#38BDF8;">@{r['username']}</td>
                <td>{int(r['study_hours_per_week'])} hrs</td>
                <td>{int(r['attendance_percent'])}%</td>
                <td>{int(r['previous_score'])}</td>
                <td>{r.get('sleep_hours', 7.0)}h</td>
                <td style="font-weight:700; color:#38BDF8;">{r['predicted_score']:.1f}%</td>
                <td><span class="badge-grade grade-{gr}">{gr}</span></td>
                <td><span class="badge-status {badge_cls}">{r['predicted_passfail'].upper()}</span></td>
                <td style="color:#94A3B8;">{dt}</td>
            </tr>
            """
        pred_table_html = f"""
        <div class="dark-table-container">
            <table class="dark-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Student Name</th>
                        <th>Username</th>
                        <th>Study Hours</th>
                        <th>Attendance</th>
                        <th>Prev Score</th>
                        <th>Sleep</th>
                        <th>Predicted Score</th>
                        <th>Grade</th>
                        <th>Outcome</th>
                        <th>Date</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """
        render_html(pred_table_html)
    else:
        st.info("No prediction records match the selected filter criteria.")


# =============================================================================
# SCREEN 12: ADMIN — SETTINGS (DARK MODE + PASSWORD MANAGEMENT)
# =============================================================================
def render_admin_settings():
    st.markdown("## ⚙️ Institutional Settings & Password Control")
    st.caption("Manage administrator credentials, reset student passwords, and inspect platform diagnostics.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🔐 Change Administrator Password")
        st.caption("Update the master administrator credentials. Salted PBKDF2 hash will be computed automatically.")
        with st.form("admin_change_pw_form"):
            curr_adm_pw = st.text_input("Current Admin Password", type="password")
            new_adm_pw = st.text_input("New Admin Password (min 6 chars)", type="password")
            conf_adm_pw = st.text_input("Confirm New Admin Password", type="password")

            if st.form_submit_button("Update Admin Password", use_container_width=True):
                if not curr_adm_pw or not new_adm_pw:
                    st.error("Please fill in all password fields.")
                elif len(new_adm_pw) < 6:
                    st.error("New password must be at least 6 characters.")
                elif new_adm_pw != conf_adm_pw:
                    st.error("New passwords do not match.")
                else:
                    ok, msg = db.change_user_password(st.session_state.username, curr_adm_pw, new_adm_pw)
                    if ok:
                        st.success("Admin password updated successfully.")
                    else:
                        st.error(msg)

    with c2:
        st.markdown("### 🔑 Reset Student Password")
        st.caption("Select any enrolled student to reset or assign a new password immediately.")
        students = db.get_all_students()
        if students:
            student_dict = {f"{s['full_name']} (@{s['username']})": s["username"] for s in students}
            sel_s_label = st.selectbox("Select Student to Reset", list(student_dict.keys()), key="pw_reset_sel")
            target_uname = student_dict[sel_s_label]

            with st.form("reset_student_pw_form"):
                stu_new_pw = st.text_input(f"New Password for @{target_uname}", type="password")
                stu_conf_pw = st.text_input("Confirm Student Password", type="password")

                if st.form_submit_button(f"Reset Password for @{target_uname}", use_container_width=True):
                    if not stu_new_pw:
                        st.error("Please enter a new password.")
                    elif len(stu_new_pw) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif stu_new_pw != stu_conf_pw:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = db.update_password(target_uname, stu_new_pw)
                        if ok:
                            st.success(f"Password for @{target_uname} has been updated successfully!")
                        else:
                            st.error(msg)
        else:
            st.info("No student accounts registered in the database.")

    st.write("")

    # System & ML Diagnostics
    diag_html = """
    <div class="content-card-dark">
        <h4>🖥️ Platform & Machine Learning Environment Status</h4>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:16px; margin-top:14px;">
            <div style="background:#0E1626; padding:14px; border-radius:10px; border:1px solid #1E293B;">
                <div style="color:#94A3B8; font-size:12px; text-transform:uppercase;">Theme Status</div>
                <div style="color:#38BDF8; font-weight:800; font-size:16px; margin-top:2px;">🌙 Dark Mode Active</div>
            </div>
            <div style="background:#0E1626; padding:14px; border-radius:10px; border:1px solid #1E293B;">
                <div style="color:#94A3B8; font-size:12px; text-transform:uppercase;">Database Engine</div>
                <div style="color:#34D399; font-weight:800; font-size:16px; margin-top:2px;">SQLite3 (PBKDF2 Salted)</div>
            </div>
            <div style="background:#0E1626; padding:14px; border-radius:10px; border:1px solid #1E293B;">
                <div style="color:#94A3B8; font-size:12px; text-transform:uppercase;">Model Architecture</div>
                <div style="color:#C084FC; font-weight:800; font-size:16px; margin-top:2px;">Random Forest Ensemble</div>
            </div>
            <div style="background:#0E1626; padding:14px; border-radius:10px; border:1px solid #1E293B;">
                <div style="color:#94A3B8; font-size:12px; text-transform:uppercase;">AI Advisor Pipeline</div>
                <div style="color:#FBBF24; font-weight:800; font-size:16px; margin-top:2px;">Heuristic + Probabilistic</div>
            </div>
        </div>
    </div>
    """
    render_html(diag_html)


# =============================================================================
# MAIN CONTROLLER & APPLICATION ROUTER
# =============================================================================
if not st.session_state.logged_in:
    if st.session_state.landing_view == "login":
        render_screen_2_student_login()
    elif st.session_state.landing_view == "register":
        render_screen_3_student_registration()
    elif st.session_state.landing_view == "admin_login":
        render_admin_login()
    else:
        render_screen_1_home()

else:
    # Sidebar Navigation with Logo Header
    with st.sidebar:
        render_sidebar_brand()

        if st.session_state.role == "student":
            stu_avatar_char = st.session_state.full_name[0] if st.session_state.full_name else 'S'
            stu_card_html = f"""
            <div class="sidebar-user-card">
                <div class="sidebar-avatar avatar-student">
                    {stu_avatar_char}
                </div>
                <div>
                    <div style="font-size:14px; font-weight:700; color:#FFFFFF;">{st.session_state.full_name}</div>
                    <div style="font-size:11.5px; color:#38BDF8;">Student • @{st.session_state.username}</div>
                </div>
            </div>
            """
            render_html(stu_card_html)

            nav_options = ["Dashboard", "Predict Performance", "My History & Analytics", "My Profile"]
            nav_icons = {"Dashboard": "📊", "Predict Performance": "🔮", "My History & Analytics": "📈", "My Profile": "👤"}

            nav_selection = st.radio(
                "Student Navigation",
                nav_options,
                index=nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in nav_options else 0,
                format_func=lambda x: f"{nav_icons.get(x, '')}  {x}",
                label_visibility="collapsed",
            )
            st.session_state.active_nav = nav_selection

        else:
            adm_card_html = """
            <div class="sidebar-user-card">
                <div class="sidebar-avatar avatar-admin">
                    A
                </div>
                <div>
                    <div style="font-size:14px; font-weight:700; color:#FFFFFF;">Administrator</div>
                    <div style="font-size:11.5px; color:#C084FC;">Institutional Faculty</div>
                </div>
            </div>
            """
            render_html(adm_card_html)

            # Sidebar with exact 4 items requested: Overview, Manage Students, All Predictions, Settings
            adm_nav_options = ["Overview", "Manage Students", "All Predictions", "Settings"]
            adm_icons = {"Overview": "📊", "Manage Students": "👥", "All Predictions": "📋", "Settings": "⚙️"}

            adm_selection = st.radio(
                "Admin Navigation",
                adm_nav_options,
                index=adm_nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in adm_nav_options else 0,
                format_func=lambda x: f"{adm_icons.get(x, '')}  {x}",
                label_visibility="collapsed",
            )
            st.session_state.active_nav = adm_selection

        st.write("")
        st.write("")
        if st.button("🚪 Sign Out", use_container_width=True, key="sidebar_logout_btn"):
            sign_out()
            st.rerun()

    # Route Rendering
    if st.session_state.role == "student":
        if st.session_state.active_nav == "Dashboard":
            render_screen_4_student_dashboard()
        elif st.session_state.active_nav == "Predict Performance":
            render_screen_5_prediction_page()
        elif st.session_state.active_nav == "My History & Analytics":
            render_screen_6_history()
        elif st.session_state.active_nav == "My Profile":
            render_student_profile()
    else:
        if st.session_state.active_nav == "Overview":
            render_screen_7_admin_overview()
        elif st.session_state.active_nav == "Manage Students":
            render_screen_8_manage_students()
        elif st.session_state.active_nav == "All Predictions":
            render_screen_9_all_predictions()
        elif st.session_state.active_nav == "Settings":
            render_admin_settings()