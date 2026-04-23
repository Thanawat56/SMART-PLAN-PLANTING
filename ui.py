import streamlit as st

def load_css():
    with open("styles.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_header():
    st.markdown("""
    <h1 class='app-title'>🌱 Smart Planting AI</h1>
    <p class='app-subtitle'>ผู้ช่วยเกษตรกรอัจฉริยะ</p>
    """)
    st.divider()

def empty_state():
    st.info("👋 สวัสดีครับ! ลองถามเกี่ยวกับ ข้าว ข้าวโพด มันสำปะหลัง อ้อย หรือถั่วเหลือง ได้เลย")