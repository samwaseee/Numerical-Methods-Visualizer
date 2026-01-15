import streamlit as st

def render_function_header(solver_instance):
    if solver_instance and solver_instance.valid:
        st.markdown("<br>", unsafe_allow_html=True)
        st.latex(r"\huge " + solver_instance.get_latex())
        st.markdown("---")