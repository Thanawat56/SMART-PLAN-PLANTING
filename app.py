import streamlit as st
import os
import warnings
import sqlite3
import html
from datetime import datetime

import ui

from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. ตั้งค่าเบื้องต้น ---
load_dotenv()
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- 2. ฟังก์ชัน Guardrails (ด่านตรวจความปลอดภัยและขอบเขตพืช) ---
def check_guardrails(text):
    text = text.lower().strip()
    # หมวดหมู่ที่ไม่เกี่ยวข้อง
    out_keywords = ["การเมือง", "นายก", "ประธานาธิบดี", "เลือกตั้ง", "ดารา", "นักร้อง", "ฟุตบอล", "หวย", "เขียนโค้ด", "โปรแกรม", 
                    "คอมพิวเตอร์", "มือถือ", "เทคโนโลยี", "วิทยาศาสตร์", "สุขภาพ", "การแพทย์", "อาหาร", "เครื่องดื่ม", "แฟชั่น", 
                    "ความงาม", "ท่องเที่ยว", "รถยนต์", "มอเตอร์ไซค์", "เกม", "หนัง", "เพลง", "ศิลปะ", "วรรณกรรม", "ประวัติศาสตร์", 
                    "ภูมิศาสตร์", "เศรษฐกิจ", "ธุรกิจ", "การเงิน", "การลงทุน", "กฎหมาย", "สิ่งแวดล้อม", "สังคม", "จิตวิทยา", "ปรัชญา", 
                    "ศาสนา", "ภาษาศาสตร์", "วัฒนธรรม","กัญชง","พืชผิดกฏหมายต" ,"อื่นๆ"]
    # หมวดหมู่พืชที่ไม่ได้รับอนุญาต (ตัวอย่าง: พริก)
    illegal_plants = ["พริก", "มะเขือเทศ", "กะเพรา", "โหระพา", "ผักชี", "ต้นหอม", "กระเทียม", "ขิง", "ข่า", "ตะไคร้", "มะนาว", "ส้ม", "ผลไม้","หูกระจง","กาแฟ","มะม่วง","แตงโม","ท้อม","ฝรั่ง","ผลไม้อื่นๆ","ผักอื่นๆ"]
    
    if any(k in text for k in out_keywords):
        return False, "ขออภัยครับ ผมเป็น AI ผู้เชี่ยวชาญด้านการเกษตร ไม่สามารถตอบคำถามในเรื่องนี้ได้ครับ"
    
    if any(p in text for p in illegal_plants):
        return False, "ขออภัยครับ ข้อมูลส่วนนี้เกินขอบเขตการตอบของผม (จำกัดเฉพาะพืช 5 ชนิดหลัก: ข้าว, ข้าวโพด, มันสำปะหลัง, อ้อย, ถั่วเหลือง)"
        
    return True, ""

# --- 3. ฟังก์ชันจัดการฐานข้อมูล SQLite (ใช้เก็บประวัติแชท) ---
def init_db():
    conn = sqlite3.connect('chat_history_langchain.db')
    c = conn.cursor()
    # ตารางเก็บ Session (หัวข้อแชท)
    c.execute('CREATE TABLE IF NOT EXISTS chat_sessions (session_id TEXT PRIMARY KEY, title TEXT, created_at DATETIME)')
    # ตารางเก็บข้อความ User/Bot
    c.execute('CREATE TABLE IF NOT EXISTS messages (session_id TEXT, role TEXT, content TEXT, timestamp DATETIME)')
    conn.commit()
    conn.close()

def save_chat_session(session_id, title):
    conn = sqlite3.connect('chat_history_langchain.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO chat_sessions VALUES (?, ?, ?)", (session_id, title, datetime.now()))
    conn.commit()
    conn.close()

def save_message(session_id, role, content):
    conn = sqlite3.connect('chat_history_langchain.db')
    c = conn.cursor()
    c.execute("INSERT INTO messages VALUES (?, ?, ?, ?)", (session_id, role, content, datetime.now()))
    conn.commit()
    conn.close()

def load_sessions():
    conn = sqlite3.connect('chat_history_langchain.db')
    c = conn.cursor()
    c.execute("SELECT session_id, title FROM chat_sessions ORDER BY created_at DESC")
    return c.fetchall()

def load_messages(session_id):
    conn = sqlite3.connect('chat_history_langchain.db')
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    return [{"role": row[0], "content": row[1]} for row in c.fetchall()]


def delete_session(session_id):
    conn = sqlite3.connect('chat_history_langchain.db')
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    c.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def load_css():
    with open("styles.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_chat_content(role, content):
    safe_content = html.escape(str(content)).replace("\n", "<br>")
    role_label = "คุณ" if role == "user" else "Smart Planting AI"
    avatar = "🧑‍🌾" if role == "user" else "🌱"
    st.markdown(
        f"""
        <div class="chat-row {role}">
            <div class="chat-bubble {role}">
                <div class='chat-role {role}'>{avatar} {role_label}</div>
                <div class='chat-text'>{safe_content}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history(messages):
    for message in messages:
        role = message.get("role", "assistant")
        render_chat_content(role, message.get("content", ""))

# --- 4. ฟังก์ชันเตรียมฐานข้อมูล RAG (LangChain) ---
@st.cache_resource
def init_rag_bot():
    data_path = "./data"
    all_docs = []
    if not os.path.exists(data_path) or not os.listdir(data_path):
        return None

    # โหลดไฟล์ทุกประเภทจากโฟลเดอร์ data
    for file in os.listdir(data_path):
        full_path = os.path.join(data_path, file)
        ext = os.path.splitext(file)[1].lower()
        try:
            if ext == ".pdf": loader = PDFPlumberLoader(full_path)
            elif ext == ".csv": loader = CSVLoader(file_path=full_path, encoding='utf-8')
            elif ext == ".docx": loader = Docx2txtLoader(full_path)
            elif ext == ".txt": loader = TextLoader(full_path, encoding='utf-8')
            else: continue
            all_docs.extend(loader.load())
        except Exception as e:
            st.error(f"ไม่สามารถโหลดไฟล์ {file} ได้: {e}")

    if not all_docs: return None

    # หั่นข้อความ (Chunking) เพื่อให้ AI ค้นหาข้อมูลได้แม่นยำ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)

    # สร้าง Vector Database (FAISS) และ Embedding (E5)
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    # ค้นหาข้อมูลที่ใกล้เคียงที่สุด 5 ส่วน (k=5)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# --- 5. UI Layout ---
# --- 5. UI Layout ---
st.set_page_config(
    page_title="SMART PLANTING AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()

# ✅ โหลด CSS (ถ้ามี ui.py)
try:
    import ui
    ui.load_css()
except:
    pass


# =========================
# 🧠 SESSION STATE
# =========================
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_delete_session_id" not in st.session_state:
    st.session_state.pending_delete_session_id = None


# =========================
# 📚 SIDEBAR (ซ้าย)
# =========================
with st.sidebar:
    st.markdown("<div class='sb-brand'>🍃 SMART PLAN PLANTING</div>", unsafe_allow_html=True)
    st.divider()

    if st.button("แชทใหม่", use_container_width=True):
        st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.messages = []
        st.rerun()


    st.divider()

    sessions = load_sessions()

    if sessions:
        for s_id, s_title in sessions:
            if st.session_state.pending_delete_session_id == s_id:
                col_open, col_confirm, col_cancel = st.columns([0.68, 0.16, 0.16], gap="small")
                with col_open:
                    if st.button(f"💬 {s_title[:20]}...", key=f"open_{s_id}", use_container_width=True):
                        st.session_state.session_id = s_id
                        st.session_state.messages = load_messages(s_id)
                        st.session_state.pending_delete_session_id = None
                        st.rerun()
                with col_confirm:
                    if st.button("🗑", key=f"confirm_delete_{s_id}", help="ยืนยันลบแชทนี้", use_container_width=True):
                        delete_session(s_id)
                        if st.session_state.session_id == s_id:
                            st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
                            st.session_state.messages = []
                        st.session_state.pending_delete_session_id = None
                        st.rerun()
                with col_cancel:
                    if st.button("✕", key=f"cancel_delete_{s_id}", help="ยกเลิก", use_container_width=True):
                        st.session_state.pending_delete_session_id = None
                        st.rerun()
            else:
                col_open, col_more = st.columns([0.82, 0.18], gap="small")
                with col_open:
                    if st.button(f"💬 {s_title[:20]}...", key=f"open_{s_id}", use_container_width=True):
                        st.session_state.session_id = s_id
                        st.session_state.messages = load_messages(s_id)
                        st.session_state.pending_delete_session_id = None
                        st.rerun()
                with col_more:
                    if st.button("⋯", key=f"menu_{s_id}", use_container_width=True):
                        st.session_state.pending_delete_session_id = s_id
                        st.rerun()
    else:
        st.caption("ยังไม่มีประวัติแชท")


# =========================
# 🌱 HEADER (ด้านบน)
# =========================
message_count = len(st.session_state.messages)
st.markdown(f"""
<div class='chatboard-hero'>
    <div class='chatboard-title-wrap'>
        <h1 class='app-title'>Chatboard</h1>
        <p class='app-subtitle'>สวัสดี ยินดีต้อนรับ</p>
    </div>
    <div class='chatboard-meta'>
        <span class='chatboard-pill'>Session: {st.session_state.session_id[-6:]}</span>
        <span class='chatboard-pill'>Messages: {message_count}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# =========================
# 💬 CHAT AREA
# =========================

# เรียกใช้งาน Bot
retriever = init_rag_bot()
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

if retriever is None:
    st.info("📌 กรุณานำไฟล์ข้อมูลไปวางในโฟลเดอร์ `data` และ Refresh หน้าจอ")
else:
    # รับคำถามจากผู้ใช้
    if query := st.chat_input("ถามคำถามเกี่ยวกับพืช 5 ชนิดหลัก..."):
        # 1. บันทึกคำถาม User ลง UI และ Database
        st.session_state.messages.append({"role": "user", "content": query})
        save_message(st.session_state.session_id, "user", query)

        # 2. ตรวจสอบ Guardrails ก่อนส่งหา AI
        is_safe, warning_msg = check_guardrails(query)

        if not is_safe:
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
            save_message(st.session_state.session_id, "assistant", warning_msg)
        else:
            with st.spinner('กำลังวิเคราะห์ข้อมูล...'):
                # ตั้งชื่อ Session ถ้าเป็นคำถามแรกของห้องแชท
                if len(st.session_state.messages) <= 1:
                    save_chat_session(st.session_state.session_id, query)

                # 3. RAG Process: ค้นหาข้อมูลจากไฟล์
                context_docs = retriever.invoke(query)
                context_text = "\n---\n".join([d.page_content for d in context_docs])
                
                # 4. Memory Re-sync: สร้างประวัติการสนทนาเพื่อส่งให้ AI (จำความหลัง)
                
                # ดึงประวัติ 5 ข้อความล่าสุดมาเรียงเป็น text
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:-1]])

                # 5. Prompt Construction
                prompt = PromptTemplate.from_template(
                   """คุณคือ 'Smart Planting AI' ผู้เชี่ยวชาญการเกษตร
                    
                    [กฎเหล็ก]
                    1. คุณให้คำปรึกษาพืชเพียง 5 ชนิดเท่านั้น: ข้าว, ข้าวโพด, มันสำปะหลัง, อ้อย, ถั่วเหลือง
                    2. ตอบโดยใช้ข้อมูลจาก [ข้อมูลอ้างอิงจากไฟล์] เป็นหลัก ห้ามแต่งข้อมูลขึ้นมาเอง
                    3. 🌟 วิธีรับมือกับคำถามกว้างๆ / ขอคำแนะนำทั่วไป:
                       - ให้วิเคราะห์ 'คำสำคัญ' ของผู้ใช้ก่อน เช่น หากผู้ใช้ถามถึงการเป็น "ชาวนา" ให้คุณเน้นแนะนำเรื่อง "ข้าว" เป็นหลัก (เพราะบริบทคนไทย ชาวนาคือผู้ปลูกข้าว) 
                       - หากถามกว้างๆ ว่าปลูกอะไรดี ให้สรุปแนะนำพืช 5 ชนิดแบบ "ภาษาพูดที่เป็นธรรมชาติ" ห้ามตอบเป็นสคริปต์ซ้ำๆ หรือก็อปปี้แพทเทิร์นเดิมมาตอบทุกครั้ง
                       - ท้ายประโยค ค่อยถามกลับอย่างเป็นธรรมชาติว่าสภาพดินหรือน้ำเป็นอย่างไร เพื่อให้คำแนะนำต่อได้
                    4. หากถามคำถามเจาะจงแล้วไม่มีข้อมูลในไฟล์ ให้ตอบว่า 'ไม่พบข้อมูลในไฟล์ครับ'
                    5. ตอบคำถามด้วยภาษาที่เป็นธรรมชาติ สรุปใจความสำคัญ และให้คำแนะนำที่ชัดเจน 
                    6. ตอบคำถาม ขั้นตอนการปลูก การดูแล และการเก็บเกี่ยวให้ครบถ้วน หากข้อมูลในไฟล์ไม่ครบ ให้ตอบตามข้อมูลที่มีและบอกว่าข้อมูลไม่ครบ

                    [ประวัติการสนทนา]
                    {history}

                    [ข้อมูลอ้างอิงจากไฟล์]
                    {context}
                    
                    คำถามปัจจุบัน: {question}
                    
                    คำตอบของคุณ (ตอบแบบธรรมชาติ 📌 สรุป และ ✅ คำแนะนำ):"""
                )
                
                # 6. ส่งข้อมูลเข้า LLM
                chain_input = prompt.format(history=history_text, context=context_text, question=query)
                response = llm.invoke(chain_input)
                
                # 7. แสดงผลและบันทึกคำตอบ Bot
                response_text = response.content.strip()
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                save_message(st.session_state.session_id, "assistant", response_text)

                # แสดงแหล่งที่มาของข้อมูล
                with st.expander("🔍 ดูแหล่งที่มา"):
                    for i, doc in enumerate(context_docs):
                        st.write(f"📄 ส่วนที่ {i+1}: {doc.metadata.get('source', 'Unknown')}")

# 🟢 Empty State
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-chat-board">
        <div class="empty-chat-icon">🍃</div>
        <div class="empty-chat-title">เริ่มสนทนาได้เลยครับ พิมพ์คำถามด้านล่าง</div>
    </div>
    """, unsafe_allow_html=True)

# 🟢 แสดง chat history
render_chat_history(st.session_state.messages)