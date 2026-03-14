"""
LearnIQ - CBSE Grade 8 Science AI Tutor
3-Panel UI: Chapter Nav | Lesson+Practice | AI Tutor Chat
Fun UI for 8th graders — streaks, XP, badges, emojis!
"""

import os
from pathlib import Path
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── CONFIG ────────────────────────────────────────────────────────────────────
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "learniq")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# ── CHAPTERS ──────────────────────────────────────────────────────────────────
CHAPTERS = [
    {"num": 1,  "title": "Crop Production and Management",    "emoji": "🌾"},
    {"num": 2,  "title": "Microorganisms: Friend and Foe",    "emoji": "🦠"},
    {"num": 3,  "title": "Synthetic Fibres and Plastics",     "emoji": "🧵"},
    {"num": 4,  "title": "Materials: Metals and Non-Metals",  "emoji": "⚗️"},
    {"num": 5,  "title": "Coal and Petroleum",                "emoji": "⛽"},
    {"num": 6,  "title": "Combustion and Flame",              "emoji": "🔥"},
    {"num": 7,  "title": "Conservation of Plants and Animals","emoji": "🌿"},
    {"num": 8,  "title": "Cell — Structure and Functions",    "emoji": "🔬"},
    {"num": 9,  "title": "Reproduction in Animals",           "emoji": "🐣"},
    {"num": 10, "title": "Reaching the Age of Adolescence",   "emoji": "🧬"},
]

CHAPTER_TOPICS = {
    1:  "crop production management irrigation manure fertilizer",
    2:  "microorganisms bacteria fungi protozoa algae curd bread",
    3:  "synthetic fibres plastics nylon rayon polyester",
    4:  "metals non-metals properties uses",
    5:  "coal petroleum natural gas fossil fuels",
    6:  "combustion flame ignition temperature fire",
    7:  "conservation plants animals deforestation wildlife",
    8:  "cell structure nucleus membrane organelles",
    9:  "reproduction animals sexual asexual",
    10: "adolescence puberty hormones changes",
}

FUN_FACTS = {
    1:  "🌾 Rice feeds more than half of the world's population every single day!",
    2:  "🦠 Your gut has more bacteria than there are stars in the Milky Way galaxy!",
    3:  "🕷️ Spider silk is stronger than steel of the same thickness — nature's own synthetic fibre!",
    4:  "⚡ Copper has been used by humans for over 10,000 years — it's one of the first metals ever used!",
    5:  "🦕 Fossil fuels are made from organisms that lived over 300 million years ago — before dinosaurs!",
    6:  "🕯️ A candle flame burns at around 1,000°C at its hottest point!",
    7:  "🐘 An elephant eats up to 150 kg of food per day to support its massive body!",
    8:  "🔬 The human body has about 37 trillion cells — all working together right now!",
    9:  "🐟 A female salmon lays up to 5,000 eggs at once to ensure some survive!",
    10: "📏 The fastest growth spurt during adolescence can be up to 10 cm in a single year!",
}

XP_PER_QUESTION = 10
XP_PER_LEVEL    = 100

# ── CHAINS ────────────────────────────────────────────────────────────────────
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

def build_qa_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_template(
        "You are LearnIQ, a fun and friendly AI tutor for CBSE Grade 8 Science students.\n\n"
        "Use ONLY the context below. Keep answers short, clear and exciting for a 13-year-old.\n"
        "Use emojis occasionally to make it fun! End with — Source: Page [number]\n\n"
        "If not found: say 'Hmm, I couldn't find that in the textbook. Try rephrasing!'\n\n"
        "Context:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return create_retrieval_chain(retriever, combine_docs_chain)

def build_lesson_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_template(
        "You are a fun textbook summariser for CBSE Grade 8 Science.\n\n"
        "Using ONLY the context below, write a clear lesson for a student. "
        "Use short paragraphs. Bold key terms. Use emojis to make it engaging! "
        "Keep it under 200 words. End with 'Source: Page X–Y'.\n\n"
        "Context:\n{context}\n\nTopic: {input}\n\nLesson summary:"
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return create_retrieval_chain(retriever, combine_docs_chain)

def build_practice_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_template(
        "You are a CBSE Grade 8 Science teacher.\n\n"
        "Using ONLY the context below, create ONE multiple-choice question.\n"
        "Format EXACTLY as:\n"
        "Q: [question]\nA) [option]\nB) [option]\nC) [option]\nD) [option]\n"
        "Answer: [correct letter]\nExplanation: [one sentence from the textbook]\n\n"
        "Context:\n{context}\n\nTopic: {input}\n\nQuestion:"
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return create_retrieval_chain(retriever, combine_docs_chain)

@st.cache_resource(show_spinner=False)
def load_all_chains(_pinecone_index: str):
    embeddings  = get_embeddings()
    vectorstore = PineconeVectorStore(index_name=_pinecone_index, embedding=embeddings)
    pc        = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
    index     = pc.Index(_pinecone_index)
    stats     = index.describe_index_stats()
    doc_count = stats.get("total_vector_count", 0)
    return (build_qa_chain(vectorstore), build_lesson_chain(vectorstore),
            build_practice_chain(vectorstore), doc_count)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def make_source_pills(source_docs):
    seen, pills = set(), []
    for doc in source_docs:
        label = doc.metadata.get("source_label", "")
        if not label:
            page  = doc.metadata.get("page", 0) + 1
            label = f"Textbook · Page {page}"
        if label not in seen:
            seen.add(label)
            pills.append(
                f'<span style="background:#ede9fe;color:#5b21b6;'
                f'padding:3px 10px;border-radius:20px;font-size:0.7rem;'
                f'margin-right:5px;display:inline-block;margin-top:4px;">'
                f'📄 {label}</span>'
            )
    return "".join(pills)

def parse_mcq(raw):
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    mcq   = {"q": "", "options": {}, "answer": "", "explanation": ""}
    for line in lines:
        if line.startswith("Q:"):
            mcq["q"] = line[2:].strip()
        elif line.startswith(("A)", "B)", "C)", "D)")):
            mcq["options"][line[0]] = line[3:].strip()
        elif line.lower().startswith("answer:"):
            mcq["answer"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("explanation:"):
            mcq["explanation"] = line.split(":", 1)[1].strip()
    return mcq

def get_level(xp):
    return xp // XP_PER_LEVEL + 1

def get_xp_progress(xp):
    return xp % XP_PER_LEVEL

# ── CSS ───────────────────────────────────────────────────────────────────────
def apply_css():
    st.markdown("""
    <style>
        .stApp { background:#fdf6ff !important; }
        .block-container { padding: 0 !important; max-width: 100% !important; }
        header[data-testid="stHeader"] {
            background:#fdf6ff !important;
            border-bottom: 1px solid #e0d4f8 !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #7c3aed 0%, #5b21b6 100%) !important;
            border-right: none !important;
        }
        section[data-testid="stSidebar"] * { color: white !important; }

        /* streak box */
        .streak-box {
            background: rgba(255,255,255,0.15); border-radius: 12px;
            padding: 10px 14px; margin: 10px 0;
            display: flex; align-items: center; gap: 10px;
        }
        .streak-val { font-size: 22px; font-weight: 900; color: #fbbf24 !important; }
        .streak-lbl { font-size: 10px; color: rgba(255,255,255,0.75) !important; }

        /* xp bar */
        .xp-track { height: 6px; background: rgba(255,255,255,0.2); border-radius: 3px; margin: 4px 0 8px; }
        .xp-fill  { height: 6px; background: #fbbf24; border-radius: 3px; }

        /* chapter items */
        .ch-item {
            display: flex; align-items: center; gap: 8px;
            padding: 7px 10px; border-radius: 8px;
            margin: 1px 0; cursor: pointer;
        }
        .ch-item:hover { background: rgba(255,255,255,0.15); }
        .ch-active { background: rgba(255,255,255,0.2) !important; }
        .ch-num {
            min-width: 22px; height: 22px; border-radius: 50%;
            background: rgba(255,255,255,0.2);
            font-size: 10px; font-weight: 700; color: white !important;
            display: flex; align-items: center; justify-content: center;
        }
        .ch-num-active { background: #fbbf24 !important; color: #5b21b6 !important; }
        .ch-label { font-size: 11px; color: rgba(255,255,255,0.85) !important; line-height: 1.3; }
        .ch-label-active { color: white !important; font-weight: 700; }

        /* badges */
        .badge-earned { font-size: 18px; }
        .badge-locked { font-size: 18px; opacity: 0.3; }

        /* center cards */
        .panel-card {
            background: #ffffff; border: 2px solid #e0d4f8;
            border-radius: 14px; padding: 20px 24px; margin-bottom: 14px;
            box-shadow: 0 2px 8px rgba(124,58,237,0.08);
        }
        .chapter-hero {
            background: linear-gradient(120deg, #7c3aed 0%, #a78bfa 100%);
            border-radius: 14px; padding: 20px 24px; margin-bottom: 14px;
            display: flex; align-items: center; gap: 16px;
        }
        .hero-icon {
            width: 52px; height: 52px; border-radius: 14px;
            background: rgba(255,255,255,0.2);
            display: flex; align-items: center; justify-content: center;
            font-size: 26px; flex-shrink: 0;
        }
        .lesson-text {
            color: #374151; font-size: 0.93rem;
            line-height: 1.85; white-space: pre-wrap;
        }
        .lesson-text b, .lesson-text strong { color: #5b21b6; }

        .fun-fact-box {
            background: linear-gradient(120deg, #fef3c7, #fde68a);
            border: 2px solid #fbbf24; border-radius: 12px;
            padding: 12px 16px; margin-top: 14px;
        }
        .fun-fact-lbl { font-size: 10px; font-weight: 800; color: #92400e;
                        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
        .fun-fact-text { font-size: 12px; color: #78350f; line-height: 1.6; }

        .xp-award {
            display: inline-flex; align-items: center; gap: 5px;
            background: #ecfdf5; border: 1.5px solid #10b981;
            color: #065f46; font-size: 11px; font-weight: 700;
            padding: 4px 10px; border-radius: 20px; margin-top: 6px;
        }

        /* chat */
        .chat-wrap { display: flex; flex-direction: column; gap: 10px; padding: 8px 0; }
        .bubble-user {
            background: #7c3aed; color: white;
            border-radius: 16px 16px 4px 16px;
            padding: 10px 14px; margin-left: auto;
            max-width: 90%; font-size: 0.85rem; line-height: 1.5;
        }
        .bubble-bot {
            background: #f5f3ff; border: 1.5px solid #e0d4f8;
            color: #3b1a8a; border-radius: 16px 16px 16px 4px;
            padding: 10px 14px; max-width: 95%;
            font-size: 0.85rem; line-height: 1.6;
        }
        .badge-row { margin-top: 8px; padding-top: 6px; border-top: 1px solid #e0d4f8; }

        /* MCQ */
        .mcq-option {
            display: flex; align-items: center; gap: 10px;
            padding: 10px 14px; border: 2px solid #e0d4f8;
            border-radius: 10px; margin-bottom: 8px;
            cursor: pointer; font-size: 13px; color: #5b21b6;
            transition: all 0.15s; background: #faf8ff;
        }
        .mcq-option:hover { border-color: #7c3aed; background: #ede9fe; }
        .mcq-correct { border-color: #10b981 !important; background: #ecfdf5 !important; color: #065f46 !important; }
        .mcq-wrong   { border-color: #ef4444 !important; background: #fef2f2 !important; color: #991b1b !important; }

        /* tutor header */
        .tutor-header {
            display: flex; align-items: center; gap: 10px;
            padding: 12px 0 10px; border-bottom: 2px solid #e0d4f8; margin-bottom: 10px;
        }
        .tutor-avatar {
            width: 36px; height: 36px; border-radius: 50%;
            background: linear-gradient(135deg, #7c3aed, #a78bfa); color: white;
            display: flex; align-items: center; justify-content: center;
            font-size: 13px; font-weight: 900; flex-shrink: 0;
        }
        .lvl-badge {
            background: #ede9fe; color: #5b21b6; font-size: 10px; font-weight: 800;
            padding: 3px 9px; border-radius: 20px; margin-left: auto;
        }

        /* suggest pills */
        .suggest-pill {
            font-size: 11px; padding: 5px 11px;
            border: 1.5px solid #c4b5fd; border-radius: 20px;
            color: #5b21b6; cursor: pointer; background: #f5f3ff;
            font-family: sans-serif; font-weight: 600; transition: all 0.15s;
        }
        .suggest-pill:hover { background: #7c3aed; color: white; border-color: #7c3aed; }

        /* inputs */
        .stTextInput input {
            background: #faf8ff !important; border: 2px solid #e0d4f8 !important;
            color: #3b1a8a !important; border-radius: 10px !important;
            font-size: 0.87rem !important;
        }
        .stTextInput input:focus { border-color: #7c3aed !important; box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important; }
        .stFormSubmitButton button {
            background: #7c3aed !important; color: white !important;
            border-radius: 10px !important; font-weight: 800 !important; border: none !important;
        }
        .stFormSubmitButton button:hover { background: #6d28d9 !important; }
        div[data-testid="stButton"] button {
            background: #f5f3ff !important; border: 1.5px solid #e0d4f8 !important;
            color: #6b4fa8 !important; border-radius: 8px !important;
            font-size: 12px !important; transition: all 0.15s !important;
        }
        div[data-testid="stButton"] button:hover {
            border-color: #7c3aed !important; color: #7c3aed !important; background: #ede9fe !important;
        }

        /* metrics */
        [data-testid="stMetricValue"] { color: #fbbf24 !important; font-size: 1.4rem !important; font-weight: 900 !important; }
        [data-testid="stMetricLabel"] { color: rgba(255,255,255,0.7) !important; font-size: 0.75rem !important; }

        /* alerts */
        div[data-testid="stAlert"] { border-radius: 10px !important; }
        .stSuccess { background: #ecfdf5 !important; color: #065f46 !important; border: 2px solid #10b981 !important; }
        .stError   { background: #fef2f2 !important; color: #991b1b !important; border: 2px solid #ef4444 !important; }
        .stSpinner > div { border-top-color: #7c3aed !important; }
        hr { border-color: #e0d4f8 !important; }
        #MainMenu, footer, .stDeployButton { display: none !important; }

        /* ── ANIMATIONS ── */
        @keyframes popIn {
            0%   { transform: scale(0) rotate(-10deg); opacity: 0; }
            70%  { transform: scale(1.15) rotate(2deg); }
            100% { transform: scale(1) rotate(0deg); opacity: 1; }
        }
        @keyframes slideInLeft {
            0%   { transform: translateX(-40px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        @keyframes bounceIn {
            0%   { transform: scale(0.8); opacity: 0; }
            50%  { transform: scale(1.05); }
            100% { transform: scale(1); opacity: 1; }
        }
        @keyframes starPop {
            0%   { transform: scale(0) rotate(-30deg); opacity: 0; }
            70%  { transform: scale(1.3) rotate(10deg); opacity: 1; }
            100% { transform: scale(1) rotate(0deg); opacity: 1; }
        }
        @keyframes fillBar {
            0%   { width: 0%; }
            100% { width: var(--xp-pct); }
        }
        @keyframes fireP { 0% { transform: scale(1); } 100% { transform: scale(1.3); } }
        @keyframes typingB { 0%,60%,100% { transform: translateY(0); } 30% { transform: translateY(-7px); } }
        @keyframes confettiF { 0% { transform: translateY(-10px) rotate(0deg); opacity:1; } 100% { transform: translateY(120px) rotate(360deg); opacity:0; } }
        @keyframes fadeSlideUp { 0% { transform: translateY(12px); opacity:0; } 100% { transform: translateY(0); opacity:1; } }

        .anim-pop     { animation: popIn 0.4s cubic-bezier(0.175,0.885,0.32,1.275) both; }
        .anim-slide   { animation: slideInLeft 0.5s cubic-bezier(0.175,0.885,0.32,1.275) both; }
        .anim-bounce  { animation: bounceIn 0.4s ease both; }
        .anim-fadeup  { animation: fadeSlideUp 0.35s ease both; }

        .xp-award {
            display: inline-flex; align-items: center; gap: 5px;
            background: #ecfdf5; border: 2px solid #10b981;
            color: #065f46; font-size: 12px; font-weight: 800;
            padding: 5px 12px; border-radius: 20px; margin-top: 6px;
            animation: popIn 0.4s cubic-bezier(0.175,0.885,0.32,1.275) both;
        }
        .level-up-banner {
            background: linear-gradient(135deg, #fbbf24, #f59e0b);
            border-radius: 14px; padding: 14px 20px;
            display: flex; align-items: center; gap: 12px;
            animation: slideInLeft 0.5s cubic-bezier(0.175,0.885,0.32,1.275) both;
            margin-bottom: 12px;
        }
        .confetti-container {
            position: relative; overflow: hidden;
            border-radius: 14px; padding: 18px 20px;
            background: linear-gradient(120deg, #ede9fe, #fdf6ff);
            border: 2px solid #c4b5fd; margin-bottom: 12px;
            text-align: center;
        }
        .confetti-dot {
            position: absolute; width: 8px; height: 8px;
            animation: confettiF linear forwards;
        }
        .typing-wrap {
            display: flex; align-items: center; gap: 5px;
            padding: 10px 16px; background: #f5f3ff;
            border: 1.5px solid #e0d4f8; border-radius: 14px 14px 14px 3px;
            width: fit-content; margin-bottom: 8px;
        }
        .typing-dot {
            width: 8px; height: 8px; border-radius: 50%; background: #7c3aed;
            animation: typingB 1.2s ease-in-out infinite;
        }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        .fire-icon { display: inline-block; animation: fireP 0.8s ease-in-out infinite alternate; }
        .star-1 { animation: starPop 0.3s ease forwards; opacity:0; }
        .star-2 { animation: starPop 0.3s 0.1s ease forwards; opacity:0; }
        .star-3 { animation: starPop 0.3s 0.2s ease forwards; opacity:0; }
        .xp-bar-animated {
            height: 14px; background: linear-gradient(90deg, #7c3aed, #a78bfa);
            border-radius: 8px; display: flex; align-items: center;
            justify-content: flex-end; padding-right: 6px;
            font-size: 10px; font-weight: 800; color: white;
            animation: fillBar 1.2s ease-out forwards;
        }
        .bubble-bot { animation: fadeSlideUp 0.35s ease both; }
        .bubble-user { animation: fadeSlideUp 0.35s ease both; }
        .panel-card { animation: fadeSlideUp 0.4s ease both; }
    </style>
    """, unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="LearnIQ", page_icon="🧪", layout="wide",
                       initial_sidebar_state="expanded")
    apply_css()

    defaults = {
        "active_chapter": 2,
        "active_tab":     "Lesson",
        "messages":       [],
        "q_count":        0,
        "xp":             0,
        "streak":         7,
        "lesson_cache":   {},
        "practice_cache": {},
        "mcq_answered":   {},
        "completed":      {1},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not OPENAI_API_KEY:
        st.error("**OPENAI_API_KEY not set.**")
        st.stop()
    if not PINECONE_API_KEY:
        st.error("**PINECONE_API_KEY not set.** Add to Streamlit Secrets.")
        st.stop()

    with st.spinner("🚀 Loading LearnIQ..."):
        qa_chain, lesson_chain, practice_chain, doc_count = load_all_chains(PINECONE_INDEX)

    ch_idx   = st.session_state.active_chapter
    ch_info  = next(c for c in CHAPTERS if c["num"] == ch_idx)
    ch_topic = CHAPTER_TOPICS.get(ch_idx, ch_info["title"])
    level    = get_level(st.session_state.xp)
    xp_prog  = get_xp_progress(st.session_state.xp)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='padding:16px 0 8px'>"
            "<span style='font-size:26px;font-weight:900;color:white;letter-spacing:-0.5px'>🧪 LearnIQ</span><br>"
            "<span style='font-size:11px;color:rgba(255,255,255,0.65);font-weight:500'>CBSE · Grade 8 · Science</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Streak
        st.markdown(
            f"<div class='streak-box'>"
            f"<span class='fire-icon' style='font-size:24px'>🔥</span>"
            f"<div><div class='streak-val'>{st.session_state.streak}</div>"
            f"<div class='streak-lbl'>day streak!</div></div>"
            f"<span style='margin-left:auto;font-size:20px'>⚡</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # XP bar
        st.markdown(
            f"<div style='font-size:10px;color:rgba(255,255,255,0.65);display:flex;justify-content:space-between'>"
            f"<span>Level {level} · {st.session_state.xp} XP</span><span>{level * XP_PER_LEVEL} XP</span></div>"
            f"<div class='xp-track'><div class='xp-bar-animated' style='--xp-pct:{xp_prog}%'>{xp_prog}%</div></div>",
            unsafe_allow_html=True,
        )

        # Chapters
        st.markdown("<div style='font-size:9px;font-weight:700;color:rgba(255,255,255,0.5);"
                    "text-transform:uppercase;letter-spacing:0.08em;padding:4px 0 3px'>Chapters</div>",
                    unsafe_allow_html=True)

        for ch in CHAPTERS:
            is_active = ch["num"] == st.session_state.active_chapter
            is_done   = ch["num"] in st.session_state.completed
            num_cls   = "ch-num ch-num-active" if is_active else "ch-num"
            lbl_cls   = "ch-label ch-label-active" if is_active else "ch-label"
            card_cls  = "ch-item ch-active" if is_active else "ch-item"
            star      = "⭐" if is_done and not is_active else ""

            st.markdown(
                f"<div class='{card_cls}'>"
                f"<div class='{num_cls}'>{ch['num']}</div>"
                f"<div class='{lbl_cls}'>{ch['emoji']} {ch['title']}</div>"
                f"<span style='margin-left:auto;font-size:12px'>{star}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Open", key=f"ch_btn_{ch['num']}", use_container_width=True):
                st.session_state.active_chapter = ch["num"]
                st.session_state.active_tab     = "Lesson"
                st.rerun()

        # Badges
        done = len(st.session_state.completed)
        st.markdown(
            "<div style='margin-top:12px;border-top:1px solid rgba(255,255,255,0.2);padding-top:10px'>"
            "<div style='font-size:9px;font-weight:700;color:rgba(255,255,255,0.5);"
            "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px'>Badges</div>"
            "<div style='display:flex;gap:6px;flex-wrap:wrap'>",
            unsafe_allow_html=True,
        )
        badge_icons = ["🏆","🔬","⚗️","🧬","🌱","🔥","💡","⭐","🎯","🚀"]
        badge_html = ""
        for i, icon in enumerate(badge_icons):
            cls = "badge-earned" if i < done else "badge-locked"
            badge_html += f"<span class='{cls}'>{icon}</span>"
        st.markdown(badge_html + "</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='border-top:1px solid rgba(255,255,255,0.2);padding-top:10px;margin-top:10px'>",
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Questions", st.session_state.q_count)
        col2.metric("XP", st.session_state.xp)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── MAIN COLUMNS ──────────────────────────────────────────────────────────
    col_center, col_right = st.columns([6, 4], gap="medium")

    # ── CENTER ────────────────────────────────────────────────────────────────
    with col_center:
        st.markdown(
            f"<div class='chapter-hero'>"
            f"<div class='hero-icon'>{ch_info['emoji']}</div>"
            f"<div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.75);font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.07em'>Chapter {ch_info['num']}</div>"
            f"<div style='font-size:19px;font-weight:900;color:white;margin:2px 0'>{ch_info['title']}</div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.7)'>NCERT Class 8 Science</div>"
            f"</div>"
            f"<div style='margin-left:auto;background:rgba(255,255,255,0.2);border-radius:20px;"
            f"padding:6px 14px;font-size:12px;font-weight:800;color:#fbbf24'>+50 XP</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        t1, t2, t3 = st.columns(3)
        if t1.button("📖  Lesson",   key="tab_lesson",   use_container_width=True):
            st.session_state.active_tab = "Lesson";   st.rerun()
        if t2.button("✏️  Practice", key="tab_practice", use_container_width=True):
            st.session_state.active_tab = "Practice"; st.rerun()
        if t3.button("📋  Summary",  key="tab_summary",  use_container_width=True):
            st.session_state.active_tab = "Summary";  st.rerun()

        active_tab = st.session_state.active_tab

        # LESSON
        if active_tab == "Lesson":
            cache_key = f"lesson_{ch_idx}"
            if cache_key not in st.session_state.lesson_cache:
                with st.spinner(f"✨ Loading lesson..."):
                    result = lesson_chain.invoke({"input": ch_topic})
                    st.session_state.lesson_cache[cache_key] = {
                        "text": result.get("answer", ""), "sources": result.get("context", [])
                    }
            lesson = st.session_state.lesson_cache[cache_key]
            st.markdown(
                f"<div class='panel-card'>"
                f"<div style='font-size:10px;font-weight:800;color:#7c3aed;text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:12px'>🔬 Lesson Content</div>"
                f"<div class='lesson-text'>{lesson['text']}</div>"
                f"<div class='badge-row'>{make_source_pills(lesson['sources'])}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            # Fun fact
            st.markdown(
                f"<div class='fun-fact-box'>"
                f"<div class='fun-fact-lbl'>🌟 Did you know?</div>"
                f"<div class='fun-fact-text'>{FUN_FACTS.get(ch_idx, '')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("✅ Mark chapter complete!", key="mark_done"):
                st.session_state.completed.add(ch_idx)
                st.session_state.xp += 50
                old_level = get_level(st.session_state.xp - 50)
                new_level = get_level(st.session_state.xp)
                if new_level > old_level:
                    st.markdown(
                        f"<div class='level-up-banner'>"
                        f"<span style='font-size:32px'>🏆</span>"
                        f"<div><div style='font-size:11px;color:#92400e;font-weight:700;text-transform:uppercase'>Level Up!</div>"
                        f"<div style='font-size:18px;font-weight:900;color:#78350f'>You reached Level {new_level}! 🎉</div>"
                        f"</div></div>",
                        unsafe_allow_html=True)
                st.markdown(
                    "<div class='confetti-container' id='confetti'>"
                    "<div style='font-size:18px;font-weight:900;color:#5b21b6'>🎉 Chapter Complete! +50 XP!</div>"
                    "<div style='margin-top:8px'>"
                    "<span class='star-1'>⭐</span> <span class='star-2'>⭐</span> <span class='star-3'>⭐</span>"
                    "</div></div>"
                    "<script>"
                    "const w=document.getElementById('confetti');"
                    "const c=['#7c3aed','#fbbf24','#10b981','#f472b6','#60a5fa'];"
                    "for(let i=0;i<20;i++){const d=document.createElement('div');"
                    "d.className='confetti-dot';"
                    "d.style.cssText=`left:${Math.random()*100}%;top:-10px;background:${c[Math.floor(Math.random()*c.length)]};"
                    "width:${6+Math.random()*6}px;height:${6+Math.random()*6}px;"
                    "border-radius:${Math.random()>.5?'50%':'2px'};"
                    "animation-duration:${0.8+Math.random()*.8}s;animation-delay:${Math.random()*.4}s;`;"
                    "w.appendChild(d);}"
                    "</script>",
                    unsafe_allow_html=True)
                st.rerun()

        # PRACTICE
        elif active_tab == "Practice":
            cache_key = f"practice_{ch_idx}"
            if cache_key not in st.session_state.practice_cache:
                with st.spinner("🎯 Generating question..."):
                    result = practice_chain.invoke({"input": ch_topic})
                    raw    = result.get("answer", "")
                    st.session_state.practice_cache[cache_key] = {
                        "raw": raw, "mcq": parse_mcq(raw), "sources": result.get("context", [])
                    }
            pdata   = st.session_state.practice_cache[cache_key]
            mcq     = pdata["mcq"]
            ans_key = f"ans_{ch_idx}"

            st.markdown(
                f"<div class='panel-card'>"
                f"<div style='font-size:10px;font-weight:800;color:#7c3aed;text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:12px'>🎯 Practice Question</div>"
                f"<div style='font-size:15px;color:#3b1a8a;font-weight:700;"
                f"margin-bottom:16px;line-height:1.5'>{mcq.get('q','')}</div>",
                unsafe_allow_html=True,
            )

            for opt_letter, opt_text in mcq.get("options", {}).items():
                answered = ans_key in st.session_state.mcq_answered
                if answered:
                    correct = mcq.get("answer", "").strip().upper()
                    chosen  = st.session_state.mcq_answered[ans_key]
                    cls = ("mcq-option mcq-correct" if opt_letter == correct
                           else "mcq-option mcq-wrong" if opt_letter == chosen
                           else "mcq-option")
                    st.markdown(
                        f"<div class='{cls}'><span style='font-weight:700'>{opt_letter})</span> {opt_text}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(f"{opt_letter})  {opt_text}", key=f"opt_{ch_idx}_{opt_letter}",
                                 use_container_width=True):
                        st.session_state.mcq_answered[ans_key] = opt_letter
                        st.rerun()

            if ans_key in st.session_state.mcq_answered:
                chosen  = st.session_state.mcq_answered[ans_key]
                correct = mcq.get("answer", "").strip().upper()
                if chosen == correct:
                    st.session_state.xp += XP_PER_QUESTION
                    st.markdown(
                        f"<div class='correct-answer anim-bounce' style='background:#ecfdf5;border:2px solid #10b981;"
                        f"border-radius:12px;padding:12px 16px;font-weight:700;color:#065f46;margin-bottom:8px'>"
                        f"🎉 Correct! {mcq.get('explanation','')}"
                        f"<div class='xp-award'>⚡ +{XP_PER_QUESTION} XP earned!</div></div>",
                        unsafe_allow_html=True)
                else:
                    st.error(f"❌ Not quite! Correct answer: **{correct}** — {mcq.get('explanation','')}")
                st.markdown(
                    f"<div class='badge-row'>{make_source_pills(pdata['sources'])}</div>",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                if c1.button("Next question 🔄", key=f"next_q_{ch_idx}"):
                    del st.session_state.practice_cache[cache_key]
                    del st.session_state.mcq_answered[ans_key]
                    st.rerun()
                if c2.button("Explain in chat 💬", key=f"explain_{ch_idx}"):
                    st.session_state.active_tab = "Lesson"
                    q = f"Explain: {mcq.get('q','')}"
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner("Searching..."):
                        res = qa_chain.invoke({"input": q})
                    st.session_state.messages.append({
                        "role": "assistant", "content": res.get("answer", ""),
                        "badges": make_source_pills(res.get("context", [])),
                    })
                    st.session_state.q_count += 1
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # SUMMARY
        elif active_tab == "Summary":
            cache_key = f"summary_{ch_idx}"
            if cache_key not in st.session_state.lesson_cache:
                with st.spinner("📋 Building summary..."):
                    result = lesson_chain.invoke({"input": f"key points summary {ch_topic}"})
                    st.session_state.lesson_cache[cache_key] = {
                        "text": result.get("answer", ""), "sources": result.get("context", [])
                    }
            summary = st.session_state.lesson_cache[cache_key]
            st.markdown(
                f"<div class='panel-card'>"
                f"<div style='font-size:10px;font-weight:800;color:#7c3aed;text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:12px'>📋 Chapter Summary</div>"
                f"<div class='lesson-text'>{summary['text']}</div>"
                f"<div class='badge-row'>{make_source_pills(summary['sources'])}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── RIGHT — CHAT ──────────────────────────────────────────────────────────
    with col_right:
        st.markdown(
            f"<div class='tutor-header'>"
            f"<div class='tutor-avatar'>AI</div>"
            f"<div>"
            f"<div style='font-size:13px;font-weight:900;color:#3b1a8a'>LearnIQ Tutor</div>"
            f"<div style='font-size:10px;color:#10b981;font-weight:700;display:flex;align-items:center;gap:4px'>"
            f"<span style='width:6px;height:6px;border-radius:50%;background:#10b981;display:inline-block'></span>Online</div>"
            f"</div>"
            f"<div class='lvl-badge'>Lv.{level}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if not st.session_state.messages:
            st.markdown(
                f"<div class='bubble-bot' style='margin-bottom:10px'>"
                f"👋 Hey! Ready to ace <b style='color:#7c3aed'>{ch_info['title']}</b>? "
                f"Ask me anything — I'm here to help! 🚀"
                f"</div>",
                unsafe_allow_html=True,
            )

        suggest_qs = {
            1:  ["What is irrigation? 💧", "Types of fertilizer? 🌱", "Crop rotation? 🔄"],
            2:  ["What is Lactobacillus? 🦠", "How do vaccines work? 💉", "What are pathogens? 🤒"],
            3:  ["What is nylon? 🧵", "Why avoid plastics? ♻️", "What is rayon? 🪡"],
            4:  ["Properties of metals? ⚙️", "What is corrosion? 🔩", "Non-metals? 🧪"],
            5:  ["How is coal formed? ⛏️", "What is petroleum? 🛢️", "Fossil fuels? 🦕"],
            6:  ["Ignition temperature? 🌡️", "Types of flame? 🕯️", "Stop a fire? 🧯"],
            7:  ["What is deforestation? 🌳", "Biosphere reserve? 🌍", "Endangered species? 🐘"],
            8:  ["What is a cell? 🔬", "Plant vs animal cell? 🌿", "What is nucleus? 🧬"],
            9:  ["What is fertilisation? 🥚", "Asexual reproduction? 🦎", "What is zygote? 🔬"],
            10: ["What is puberty? 📏", "Role of hormones? 🧪", "What is adolescence? 🎒"],
        }
        suggests = suggest_qs.get(ch_idx, ["Ask me anything! 🤔"])
        cols_s = st.columns(len(suggests))
        for i, sq in enumerate(suggests):
            if cols_s[i].button(sq, key=f"sq_{ch_idx}_{i}"):
                st.session_state.messages.append({"role": "user", "content": sq})
                st.session_state.q_count += 1
                st.session_state.xp += XP_PER_QUESTION
                with st.spinner("🔍 Searching..."):
                    res = qa_chain.invoke({"input": sq})
                st.session_state.messages.append({
                    "role": "assistant", "content": res.get("answer", ""),
                    "badges": make_source_pills(res.get("context", [])),
                })
                st.rerun()

        st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                xp_tag = f'<div class="xp-award">⚡ +{XP_PER_QUESTION} XP earned!</div>'
                st.markdown(
                    f'<div class="bubble-bot">{msg["content"]}'
                    f'<div class="badge-row">{msg.get("badges","")}</div>'
                    f'{xp_tag}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            c1, c2 = st.columns([8, 2])
            question = c1.text_input("q", label_visibility="collapsed",
                                     placeholder="Ask anything from the textbook...")
            go = c2.form_submit_button("Ask ➤")

        if go and question.strip():
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.q_count += 1
            st.session_state.xp += XP_PER_QUESTION
            st.markdown("<div class='typing-wrap'><div class='typing-dot'></div>"
                        "<div class='typing-dot'></div><div class='typing-dot'></div></div>",
                        unsafe_allow_html=True)
            with st.spinner("🔍 Searching textbook..."):
                result = qa_chain.invoke({"input": question})
            st.session_state.messages.append({
                "role": "assistant", "content": result.get("answer", ""),
                "badges": make_source_pills(result.get("context", [])),
            })
            st.rerun()

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
