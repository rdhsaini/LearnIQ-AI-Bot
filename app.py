"""
LearnIQ - CBSE Grade 8 Science AI Tutor
3-Panel UI: Chapter Nav | Lesson+Practice | AI Tutor Chat
Built on existing: LangChain + ChromaDB + GPT-4o-mini

RUN ORDER:
  1. python ingest.py        (once — builds ./chroma_db/)
  2. streamlit run app.py    (every time after)
"""

import os
from pathlib import Path
import streamlit as st
from streamlit_chat import message   # pip install streamlit-chat

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHROMA_DIR     = "./chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── NCERT CLASS 8 CHAPTERS ────────────────────────────────────────────────────
CHAPTERS = [
    {"num": 1,  "title": "Crop Production and Management"},
    {"num": 2,  "title": "Microorganisms: Friend and Foe"},
    {"num": 3,  "title": "Synthetic Fibres and Plastics"},
    {"num": 4,  "title": "Materials: Metals and Non-Metals"},
    {"num": 5,  "title": "Coal and Petroleum"},
    {"num": 6,  "title": "Combustion and Flame"},
    {"num": 7,  "title": "Conservation of Plants and Animals"},
    {"num": 8,  "title": "Cell — Structure and Functions"},
    {"num": 9,  "title": "Reproduction in Animals"},
    {"num": 10, "title": "Reaching the Age of Adolescence"},
]

# Key topics per chapter — used to seed the lesson view via RAG
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


# ── EMBEDDINGS + CHAIN ────────────────────────────────────────────────────────
def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )


def build_qa_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    prompt = ChatPromptTemplate.from_template(
        "You are LearnIQ, an AI tutor for CBSE Grade 8 Science.\n\n"
        "Use ONLY the context below. Do NOT use outside knowledge.\n\n"
        "If the answer exists: give a clear student-friendly explanation "
        "and end with — Source: Page [number]\n\n"
        "If not found: say 'I could not find this in the textbook. "
        "Please rephrase your question.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n\nAnswer:"
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    return create_retrieval_chain(retriever, combine_docs_chain)


def build_lesson_chain(vectorstore):
    """Separate chain for fetching lesson content — returns more chunks."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    prompt = ChatPromptTemplate.from_template(
        "You are a textbook summariser for CBSE Grade 8 Science.\n\n"
        "Using ONLY the context below, write a clear lesson summary "
        "for a student. Use short paragraphs. Bold key terms. "
        "Keep it under 200 words. End with 'Source: Page X–Y'.\n\n"
        "Context:\n{context}\n\n"
        "Topic: {input}\n\nLesson summary:"
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )
    return create_retrieval_chain(retriever, combine_docs_chain)


def build_practice_chain(vectorstore):
    """Chain that generates a practice MCQ from retrieved context."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY,
    )
    prompt = ChatPromptTemplate.from_template(
        "You are a CBSE Grade 8 Science teacher.\n\n"
        "Using ONLY the context below, create ONE multiple-choice question.\n"
        "Format EXACTLY as:\n"
        "Q: [question]\n"
        "A) [option]\n"
        "B) [option]\n"
        "C) [option]\n"
        "D) [option]\n"
        "Answer: [correct letter]\n"
        "Explanation: [one sentence from the textbook]\n\n"
        "Context:\n{context}\n\n"
        "Topic: {input}\n\nQuestion:"
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    return create_retrieval_chain(retriever, combine_docs_chain)


@st.cache_resource(show_spinner=False)
def load_all_chains(chroma_dir: str):
    embeddings  = get_embeddings()
    vectorstore = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )
    return (
        build_qa_chain(vectorstore),
        build_lesson_chain(vectorstore),
        build_practice_chain(vectorstore),
        vectorstore._collection.count(),
    )


# ── HELPERS ───────────────────────────────────────────────────────────────────
def make_source_pills(source_docs: list) -> str:
    seen, pills = set(), []
    for doc in source_docs:
        label = doc.metadata.get("source_label", "")
        if not label:
            page  = doc.metadata.get("page", 0) + 1
            label = f"Textbook · Page {page}"
        if label not in seen:
            seen.add(label)
            pills.append(
                f'<span style="background:#312e81;color:#a5b4fc;'
                f'padding:3px 10px;border-radius:20px;font-size:0.7rem;'
                f'margin-right:5px;display:inline-block;margin-top:4px;">'
                f'📄 {label}</span>'
            )
    return "".join(pills)


def parse_mcq(raw: str):
    """Parse the MCQ string into structured dict."""
    lines  = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    mcq    = {"q": "", "options": {}, "answer": "", "explanation": ""}
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


# ── CSS — Slate + Indigo palette ─────────────────────────────────────────────
# Background:  #0f1117  (deep navy-slate)
# Surface:     #1a1d2e  (elevated panels)
# Border:      #252840  (subtle dividers)
# Accent:      #4f46e5  (indigo — buttons, active states)
# Accent soft: #818cf8  (indigo-300 — hover, highlights)
# Text pri:    #e2e8f0  (near white)
# Text sec:    #8892a4  (muted slate)
# Text muted:  #4a5568  (very muted)
# Success:     #10b981  (emerald)
# Error:       #ef4444  (red)
def apply_css():
    st.markdown("""
    <style>
        /* ── Global ── */
        .stApp { background:#0f1117 !important; }
        .block-container { padding: 0 !important; max-width: 100% !important; }
        header[data-testid="stHeader"] { background:#0f1117 !important; }
        section[data-testid="stSidebar"] {
            background:#0f1117 !important;
            border-right: 1px solid #252840 !important;
        }

        /* ── Sidebar chapter list ── */
        .ch-item {
            display: flex; align-items: center; gap: 10px;
            padding: 8px 12px; border-radius: 8px; cursor: pointer;
            margin-bottom: 2px; transition: background 0.15s;
        }
        .ch-item:hover { background: #1a1d2e; }
        .ch-active {
            background: #1e1b4b !important;
            border-left: 3px solid #4f46e5;
        }
        .ch-num {
            min-width: 24px; height: 24px; border-radius: 50%;
            background: #252840; color: #8892a4;
            font-size: 11px; font-weight: 600;
            display: flex; align-items: center; justify-content: center;
        }
        .ch-num-active { background: #4f46e5 !important; color: white !important; }
        .ch-label { font-size: 12px; color: #8892a4; line-height: 1.3; }
        .ch-label-active { color: #e2e8f0 !important; font-weight: 500; }
        .ch-done {
            width: 8px; height: 8px; border-radius: 50%;
            background: #10b981; margin-left: auto; flex-shrink: 0;
        }

        /* ── Progress bar ── */
        .prog-track {
            height: 4px; background: #252840;
            border-radius: 2px; margin: 8px 0 4px;
        }
        .prog-fill {
            height: 4px; background: #4f46e5;
            border-radius: 2px; transition: width 0.3s;
        }

        /* ── Center panel ── */
        .panel-card {
            background: #1a1d2e; border: 1px solid #252840;
            border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
        }
        .chapter-hero {
            background: #1e1b4b;
            border: 1px solid #312e81; border-radius: 12px;
            padding: 20px 24px; margin-bottom: 16px;
        }
        .lesson-text {
            color: #c7d2fe; font-size: 0.92rem;
            line-height: 1.8; white-space: pre-wrap;
        }
        .lesson-text b, .lesson-text strong { color: #e2e8f0; }

        /* ── Tab bar ── */
        .tab-bar {
            display: flex; gap: 4px; margin-bottom: 16px;
            border-bottom: 1px solid #252840; padding-bottom: 0;
        }
        .tab-btn {
            padding: 8px 18px; border-radius: 8px 8px 0 0;
            border: none; background: transparent;
            color: #8892a4; font-size: 13px; cursor: pointer;
            border-bottom: 2px solid transparent;
            font-family: sans-serif; transition: all 0.15s;
        }
        .tab-btn:hover { color: #e2e8f0; }
        .tab-btn-active {
            color: #818cf8 !important;
            border-bottom-color: #4f46e5 !important;
            font-weight: 600;
        }

        /* ── Chat bubbles ── */
        .chat-wrap {
            display: flex; flex-direction: column; gap: 10px; padding: 10px 0;
        }
        .bubble-user {
            background: #4f46e5; color: #e0e7ff;
            border-radius: 16px 16px 4px 16px;
            padding: 10px 14px; margin-left: auto;
            max-width: 90%; font-size: 0.85rem; line-height: 1.5;
        }
        .bubble-bot {
            background: #1a1d2e; border: 1px solid #252840;
            color: #c7d2fe; border-radius: 16px 16px 16px 4px;
            padding: 10px 14px; max-width: 95%;
            font-size: 0.85rem; line-height: 1.6;
        }
        .badge-row {
            margin-top: 8px; padding-top: 6px;
            border-top: 1px solid #252840;
        }

        /* ── MCQ options ── */
        .mcq-option {
            display: flex; align-items: center; gap: 10px;
            padding: 9px 12px; border: 1px solid #252840;
            border-radius: 8px; margin-bottom: 6px;
            cursor: pointer; font-size: 13px; color: #8892a4;
            transition: all 0.15s; background: #0f1117;
        }
        .mcq-option:hover {
            border-color: #4f46e5; color: #e2e8f0; background: #1e1b4b;
        }
        .mcq-correct {
            border-color: #10b981 !important;
            background: #064e3b !important; color: #6ee7b7 !important;
        }
        .mcq-wrong {
            border-color: #ef4444 !important;
            background: #450a0a !important; color: #fca5a5 !important;
        }

        /* ── Tutor panel header ── */
        .tutor-header {
            display: flex; align-items: center; gap: 10px;
            padding: 12px 0 10px; border-bottom: 1px solid #252840;
            margin-bottom: 10px;
        }
        .tutor-avatar {
            width: 32px; height: 32px; border-radius: 50%;
            background: #4f46e5; color: #e0e7ff;
            display: flex; align-items: center; justify-content: center;
            font-size: 13px; font-weight: 600; flex-shrink: 0;
        }

        /* ── Suggest pills ── */
        .suggest-row { display: flex; flex-wrap: wrap; gap: 5px; margin: 6px 0 10px; }
        .suggest-pill {
            font-size: 11px; padding: 4px 10px;
            border: 1px solid #252840; border-radius: 20px;
            color: #8892a4; cursor: pointer; background: transparent;
            font-family: sans-serif; transition: all 0.15s;
        }
        .suggest-pill:hover { border-color: #818cf8; color: #818cf8; }

        /* ── Inputs ── */
        .stTextInput input {
            background: #1a1d2e !important; border: 1px solid #252840 !important;
            color: #e2e8f0 !important; border-radius: 10px !important;
            font-size: 0.85rem !important;
        }
        .stTextInput input:focus {
            border-color: #4f46e5 !important;
            box-shadow: 0 0 0 2px #312e8140 !important;
        }
        .stFormSubmitButton button {
            background: #4f46e5 !important; color: #e0e7ff !important;
            border-radius: 10px !important; font-weight: 600 !important;
            border: none !important; transition: background 0.15s !important;
        }
        .stFormSubmitButton button:hover {
            background: #4338ca !important;
        }
        div[data-testid="stButton"] button {
            background: transparent !important; border: 1px solid #252840 !important;
            color: #8892a4 !important; border-radius: 8px !important;
            font-size: 12px !important; transition: all 0.15s !important;
        }
        div[data-testid="stButton"] button:hover {
            border-color: #4f46e5 !important; color: #818cf8 !important;
            background: #1e1b4b !important;
        }

        /* ── Metrics ── */
        [data-testid="stMetricValue"] { color: #818cf8 !important; font-size: 1.2rem !important; }
        [data-testid="stMetricLabel"] { color: #8892a4 !important; font-size: 0.75rem !important; }

        /* ── Alerts / success / error ── */
        div[data-testid="stAlert"] { border-radius: 10px !important; }
        .stSuccess { background: #064e3b !important; color: #6ee7b7 !important;
                     border: 1px solid #10b981 !important; }
        .stError   { background: #450a0a !important; color: #fca5a5 !important;
                     border: 1px solid #ef4444 !important; }

        /* ── Spinner ── */
        .stSpinner > div { border-top-color: #4f46e5 !important; }

        /* ── Hide Streamlit chrome ── */
        #MainMenu, footer, .stDeployButton { display: none !important; }
    </style>
    """, unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="LearnIQ",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_css()

    # ── Session state init ────────────────────────────────────────────────────
    defaults = {
        "active_chapter": 2,
        "active_tab":     "Lesson",
        "messages":       [],
        "q_count":        0,
        "lesson_cache":   {},
        "practice_cache": {},
        "mcq_answered":   {},
        "completed":      {1},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Guards ────────────────────────────────────────────────────────────────
    if not OPENAI_API_KEY:
        st.error("**OPENAI_API_KEY not set.** Add to `.env`: `OPENAI_API_KEY=sk-...`")
        st.stop()
    if not Path(CHROMA_DIR).exists() or not any(Path(CHROMA_DIR).iterdir()):
        st.error("**Run `python ingest.py` first** to build the knowledge base.")
        st.stop()

    # ── Load chains ───────────────────────────────────────────────────────────
    with st.spinner("Loading knowledge base..."):
        qa_chain, lesson_chain, practice_chain, doc_count = load_all_chains(CHROMA_DIR)

    ch_idx   = st.session_state.active_chapter
    ch_info  = next(c for c in CHAPTERS if c["num"] == ch_idx)
    ch_topic = CHAPTER_TOPICS.get(ch_idx, ch_info["title"])

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT SIDEBAR — Chapter navigation
    # ══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown(
            "<div style='padding:14px 0 6px'>"
            "<span style='font-size:18px;font-weight:600;color:#e2e8f0'>🧪 LearnIQ</span><br>"
            "<span style='font-size:11px;color:#8b949e'>CBSE · Grade 8 · Science</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        done = len(st.session_state.completed)
        pct  = int(done / len(CHAPTERS) * 100)
        st.markdown(
            f"<div style='font-size:10px;color:#8b949e;display:flex;"
            f"justify-content:space-between'>"
            f"<span>Progress</span><span>{done}/{len(CHAPTERS)} chapters</span></div>"
            f"<div class='prog-track'><div class='prog-fill' style='width:{pct}%'></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='font-size:10px;color:#8b949e;padding:10px 0 4px;"
                    "text-transform:uppercase;letter-spacing:0.05em'>Chapters</div>",
                    unsafe_allow_html=True)

        for ch in CHAPTERS:
            is_active = ch["num"] == st.session_state.active_chapter
            is_done   = ch["num"] in st.session_state.completed
            num_cls   = "ch-num ch-num-active" if is_active else "ch-num"
            lbl_cls   = "ch-label ch-label-active" if is_active else "ch-label"
            card_cls  = "ch-item ch-active" if is_active else "ch-item"
            done_dot  = "<div class='ch-done'></div>" if is_done and not is_active else ""

            st.markdown(
                f"<div class='{card_cls}' id='ch{ch['num']}'>"
                f"<div class='{num_cls}'>{ch['num']}</div>"
                f"<div class='{lbl_cls}'>{ch['title']}</div>"
                f"{done_dot}</div>",
                unsafe_allow_html=True,
            )
            if st.button(f"Open", key=f"ch_btn_{ch['num']}",
                         help=ch["title"], use_container_width=True):
                st.session_state.active_chapter = ch["num"]
                st.session_state.active_tab     = "Lesson"
                st.rerun()

        st.markdown("<div style='margin-top:16px;border-top:1px solid #21262d;"
                    "padding-top:12px'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Questions", st.session_state.q_count)
        col2.metric("Cost", f"${st.session_state.q_count * 0.00018:.3f}")
        st.caption(f"✅ {doc_count} chunks · Budget $5.00")
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN AREA — two columns: center (lesson) + right (chat)
    # ══════════════════════════════════════════════════════════════════════════
    col_center, col_right = st.columns([6, 4], gap="medium")

    # ══════════════════════════════════════════════════════════════════════════
    # CENTER COLUMN — Chapter hero + Tabs (Lesson / Practice / Summary)
    # ══════════════════════════════════════════════════════════════════════════
    with col_center:
        # Chapter hero
        st.markdown(
            f"<div class='chapter-hero'>"
            f"<div style='font-size:11px;color:#8b949e;margin-bottom:4px'>"
            f"Chapter {ch_info['num']}</div>"
            f"<div style='font-size:18px;font-weight:600;color:#e2e8f0;margin-bottom:4px'>"
            f"{ch_info['title']}</div>"
            f"<div style='font-size:11px;color:#8b949e'>"
            f"NCERT Class 8 Science</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Tab selector
        t1, t2, t3 = st.columns(3)
        if t1.button("📖  Lesson",    key="tab_lesson",
                     use_container_width=True):
            st.session_state.active_tab = "Lesson"
            st.rerun()
        if t2.button("✏️  Practice",  key="tab_practice",
                     use_container_width=True):
            st.session_state.active_tab = "Practice"
            st.rerun()
        if t3.button("📋  Summary",   key="tab_summary",
                     use_container_width=True):
            st.session_state.active_tab = "Summary"
            st.rerun()

        active_tab = st.session_state.active_tab

        # ── TAB: LESSON ───────────────────────────────────────────────────────
        if active_tab == "Lesson":
            cache_key = f"lesson_{ch_idx}"
            if cache_key not in st.session_state.lesson_cache:
                with st.spinner(f"Loading lesson for Chapter {ch_idx}..."):
                    result = lesson_chain.invoke({"input": ch_topic})
                    st.session_state.lesson_cache[cache_key] = {
                        "text":    result.get("answer", ""),
                        "sources": result.get("context", []),
                    }

            lesson = st.session_state.lesson_cache[cache_key]
            st.markdown(
                f"<div class='panel-card'>"
                f"<div style='font-size:11px;color:#8b949e;margin-bottom:10px'>"
                f"LESSON CONTENT</div>"
                f"<div class='lesson-text'>{lesson['text']}</div>"
                f"<div class='badge-row'>{make_source_pills(lesson['sources'])}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Mark chapter as complete ✓", key="mark_done"):
                st.session_state.completed.add(ch_idx)
                st.success("Chapter marked complete!")
                st.rerun()

        # ── TAB: PRACTICE ─────────────────────────────────────────────────────
        elif active_tab == "Practice":
            cache_key = f"practice_{ch_idx}"
            if cache_key not in st.session_state.practice_cache:
                with st.spinner("Generating practice question..."):
                    result = practice_chain.invoke({"input": ch_topic})
                    raw    = result.get("answer", "")
                    st.session_state.practice_cache[cache_key] = {
                        "raw":     raw,
                        "mcq":     parse_mcq(raw),
                        "sources": result.get("context", []),
                    }

            pdata  = st.session_state.practice_cache[cache_key]
            mcq    = pdata["mcq"]
            ans_key = f"ans_{ch_idx}"

            st.markdown(
                f"<div class='panel-card'>"
                f"<div style='font-size:11px;color:#8b949e;margin-bottom:10px'>"
                f"PRACTICE QUESTION</div>"
                f"<div style='font-size:14px;color:#e2e8f0;font-weight:500;"
                f"margin-bottom:14px;line-height:1.5'>{mcq.get('q','')}</div>",
                unsafe_allow_html=True,
            )

            for opt_letter, opt_text in mcq.get("options", {}).items():
                answered = ans_key in st.session_state.mcq_answered
                if answered:
                    correct = mcq.get("answer", "").strip().upper()
                    chosen  = st.session_state.mcq_answered[ans_key]
                    if opt_letter == correct:
                        cls = "mcq-option mcq-correct"
                    elif opt_letter == chosen:
                        cls = "mcq-option mcq-wrong"
                    else:
                        cls = "mcq-option"
                    st.markdown(
                        f"<div class='{cls}'>"
                        f"<span style='font-weight:600'>{opt_letter})</span> {opt_text}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(f"{opt_letter})  {opt_text}",
                                 key=f"opt_{ch_idx}_{opt_letter}",
                                 use_container_width=True):
                        st.session_state.mcq_answered[ans_key] = opt_letter
                        st.rerun()

            if ans_key in st.session_state.mcq_answered:
                chosen  = st.session_state.mcq_answered[ans_key]
                correct = mcq.get("answer", "").strip().upper()
                if chosen == correct:
                    st.success(f"✅ Correct! {mcq.get('explanation','')}")
                else:
                    st.error(
                        f"❌ Incorrect. Correct answer: **{correct}**\n\n"
                        f"{mcq.get('explanation','')}"
                    )
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
                    with st.spinner("Searching textbook..."):
                        res = qa_chain.invoke({"input": q})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": res.get("answer", ""),
                        "badges": make_source_pills(res.get("context", [])),
                    })
                    st.session_state.q_count += 1
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # ── TAB: SUMMARY ──────────────────────────────────────────────────────
        elif active_tab == "Summary":
            cache_key = f"summary_{ch_idx}"
            if cache_key not in st.session_state.lesson_cache:
                with st.spinner("Building chapter summary..."):
                    result = lesson_chain.invoke({
                        "input": f"key points summary {ch_topic}"
                    })
                    st.session_state.lesson_cache[cache_key] = {
                        "text":    result.get("answer", ""),
                        "sources": result.get("context", []),
                    }
            summary = st.session_state.lesson_cache[cache_key]
            st.markdown(
                f"<div class='panel-card'>"
                f"<div style='font-size:11px;color:#8b949e;margin-bottom:10px'>"
                f"CHAPTER SUMMARY</div>"
                f"<div class='lesson-text'>{summary['text']}</div>"
                f"<div class='badge-row'>{make_source_pills(summary['sources'])}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN — AI Tutor Chat (persistent, always visible)
    # ══════════════════════════════════════════════════════════════════════════
    with col_right:
        # Tutor header
        st.markdown(
            "<div class='tutor-header'>"
            "<div class='tutor-avatar'>AI</div>"
            "<div>"
            "<div style='font-size:13px;font-weight:600;color:#e2e8f0'>LearnIQ Tutor</div>"
            "<div style='font-size:10px;color:#16a34a;display:flex;align-items:center;gap:4px'>"
            "<span style='width:6px;height:6px;border-radius:50%;"
            "background:#16a34a;display:inline-block'></span>Online</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # Welcome message if no chat
        if not st.session_state.messages:
            st.markdown(
                "<div class='bubble-bot' style='margin-bottom:10px'>"
                f"Hi! I'm your LearnIQ tutor. Ask me anything about "
                f"<b>{ch_info['title']}</b> or any chapter from your textbook."
                "</div>",
                unsafe_allow_html=True,
            )

        # Suggested questions
        suggest_qs = {
            1:  ["What is irrigation?", "Types of fertilizer?", "What is crop rotation?"],
            2:  ["What is Lactobacillus?", "How do vaccines work?", "What are pathogens?"],
            3:  ["What is nylon made of?", "Why avoid plastics?", "What is rayon?"],
            4:  ["Properties of metals?", "What is corrosion?", "Non-metal examples?"],
            5:  ["How is coal formed?", "What is petroleum?", "What are fossil fuels?"],
            6:  ["What is ignition temperature?", "Types of flame?", "How to stop fire?"],
            7:  ["What is deforestation?", "What is a biosphere reserve?", "Endangered species?"],
            8:  ["What is a cell?", "Difference: plant vs animal cell?", "What is nucleus?"],
            9:  ["What is fertilisation?", "Asexual reproduction examples?", "What is a zygote?"],
            10: ["What is puberty?", "Role of hormones?", "What is adolescence?"],
        }
        suggests = suggest_qs.get(ch_idx, ["Ask me anything..."])
        cols_s = st.columns(len(suggests))
        for i, sq in enumerate(suggests):
            if cols_s[i].button(sq, key=f"sq_{ch_idx}_{i}"):
                st.session_state.messages.append({"role": "user", "content": sq})
                st.session_state.q_count += 1
                with st.spinner("Searching..."):
                    res = qa_chain.invoke({"input": sq})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res.get("answer", ""),
                    "badges": make_source_pills(res.get("context", [])),
                })
                st.rerun()

        # Chat messages
        st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="bubble-user">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="bubble-bot">{msg["content"]}'
                    f'<div class="badge-row">{msg.get("badges","")}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            c1, c2 = st.columns([8, 2])
            question = c1.text_input(
                "q", label_visibility="collapsed",
                placeholder="Ask anything from the textbook...",
            )
            go = c2.form_submit_button("Ask ➤")

        if go and question.strip():
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.q_count += 1
            with st.spinner("Searching textbook..."):
                result = qa_chain.invoke({"input": question})
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("answer", ""),
                "badges": make_source_pills(result.get("context", [])),
            })
            st.rerun()

        # Clear chat
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
