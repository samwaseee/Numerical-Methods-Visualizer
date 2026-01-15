import base64
import streamlit as st

def render_navbar():
    svg_code = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="none">
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#4EA5FF;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#A855F7;stop-opacity:1" />
            </linearGradient>
        </defs>
        <path d="M 10 50 C 20 10, 30 10, 50 50 S 80 90, 90 50" stroke="url(#grad1)" stroke-width="8" stroke-linecap="round" fill="none"/>
        <circle cx="50" cy="50" r="8" fill="#FF4B4B"/>
        <line x1="5" y1="50" x2="95" y2="50" stroke="white" stroke-width="2" opacity="0.5"/>
    </svg>
    """
    b64_logo = base64.b64encode(svg_code.encode("utf-8")).decode("utf-8")
    
    st.markdown(f"""
        <style>
            /* Fixed Navbar at the top */
            .navbar {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 70px;
                background-color: #0E1117;
                z-index: 99999;
                border-bottom: 1px solid #333;
                display: flex;
                align-items: center;
                padding-left: 20px;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            }}
            .navbar img {{ height: 45px; margin-right: 15px; }}
            .navbar-title {{
                font-family: 'Source Sans Pro', sans-serif;
                font-size: 1.6rem;
                font-weight: 700;
                color: white;
            }}
            .main .block-container {{ padding-top: 80px !important; }}
            
            /* Make header transparent so 3-dots are visible */
            header[data-testid="stHeader"] {{
                background: transparent;
                z-index: 100000;
            }}
            
            footer {{ visibility: hidden; }}
            
            /* Professional Table Styling */
            div[data-testid="stDataFrame"] {{ border: 1px solid #333; border-radius: 5px; }}
        </style>
        
        <div class="navbar">
            <img src="data:image/svg+xml;base64,{b64_logo}">
            <div class="navbar-title">Numerical Methods Visualizer</div>
        </div>
    """, unsafe_allow_html=True)
