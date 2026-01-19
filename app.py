import streamlit as st
from components.navbar import render_navbar
import views.input_page as input_view
import views.result_page as result_view
import views.dashboard_page as dashboard_view

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Numerical Methods Visualizer", page_icon="📈")

# --- 2. BROWSER STORAGE ENGINE ---
qp = st.query_params

def get_from_url(key, default, cast_func=str):
    """Safely gets a value from the URL. Treats empty strings as Default."""
    if key in qp:
        try:
            val = cast_func(qp[key])
            return val if val else default 
        except:
            return default
    return default

# --- 3. INITIALIZE STATE (Load from URL or Defaults) ---
if 'page' not in st.session_state:
    st.session_state.page = get_from_url("page", "input")

# Inputs (with robust defaults)
if 'func_str' not in st.session_state: 
    st.session_state.func_str = get_from_url("func", "x^4 + 6*x^3 - 77*x^2 - 330*x + 400")

if 'x0_val' not in st.session_state: 
    st.session_state.x0_val = get_from_url("x0", 3.0, float)

if 'range_a' not in st.session_state: 
    st.session_state.range_a = get_from_url("a", -10.0, float)

if 'range_b' not in st.session_state: 
    st.session_state.range_b = get_from_url("b", 10.0, float)

if 'g_str' not in st.session_state: 
    url_g = get_from_url("g", "")
    if url_g:
        st.session_state.g_str = url_g
    else:
        st.session_state.g_str = f"x - ({st.session_state.func_str})/1000"

if 'scanner_step' not in st.session_state:
    st.session_state.scanner_step = get_from_url("step", 0.15, float)

if 'solver' not in st.session_state: st.session_state.solver = None
if 'bench_data' not in st.session_state: st.session_state.bench_data = None

# --- 4. NAVIGATION ROUTER ---
render_navbar()
header_container = st.empty()

current_page = st.session_state.page

if current_page == "input":
    input_view.show_input_page(header_container)
elif current_page == "result":
    result_view.show_result_page(header_container)
elif current_page == "dashboard":
    dashboard_view.show_dashboard_page(header_container)

# Sync Page to URL
if st.query_params.get("page") != current_page:
    st.query_params["page"] = current_page