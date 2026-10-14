"""
generate_project_pdf.py – Generates a comprehensive, professional PDF documentation
for the AI Study Assistant RAG project.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfgen import canvas

# ── NumberedCanvas for "Page X of Y" and running header/footer ─────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Don't draw header on page 1
        if self._pageNumber > 1:
            self.drawString(54, 750, "AI Study Assistant -- Production RAG Architecture & Engineering Guide")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer on all pages
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(558, 36, page_str)
        self.drawString(54, 36, "Confidential -- Portfolio Project Documentation | AI/ML Engineering")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)

        self.restoreState()


def build_pdf(output_filename: str):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom color palette
    c_primary = colors.HexColor("#1A365D")    # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0")  # Slate Blue
    c_dark = colors.HexColor("#2D3748")       # Charcoal text
    c_light = colors.HexColor("#F7FAFC")      # Light background
    c_border = colors.HexColor("#CBD5E0")

    # Typography styles
    styles.add(ParagraphStyle(
        name="DocTitle",
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=27,
        textColor=c_primary,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="DocSubtitle",
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=4,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="SubSectionHeader",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="BodyCustom",
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.5,
        textColor=c_dark,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="BodyBold",
        fontName="Helvetica-Bold",
        fontSize=8.8,
        leading=12.5,
        textColor=c_dark,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="BulletCustom",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=c_dark,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="TableHead",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=c_dark,
    ))
    styles.add(ParagraphStyle(
        name="TableCellBold",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=c_dark,
    ))

    story = []

    # ── HEADER & META ──────────────────────────────────────────────────────────
    story.append(Paragraph("AI Study Assistant", styles["DocTitle"]))
    story.append(Paragraph("Production-Grade Offline & Cloud-Resilient RAG System -- Architecture & Technical Guide", styles["DocSubtitle"]))
    
    meta_data = [
        [
            Paragraph("<b>Target Role:</b> AI/ML Engineering Portfolio", styles["TableCell"]),
            Paragraph("<b>Test Suite:</b> 67 / 67 Passed (100% Pass Rate)", styles["TableCell"]),
        ],
        [
            Paragraph("<b>Stack:</b> Python 3.10+, FAISS, Streamlit, Ollama", styles["TableCell"]),
            Paragraph("<b>Default Offline Model:</b> Qwen2.5-Coder:7B / Llama 3.2:3B", styles["TableCell"]),
        ]
    ]
    t_meta = Table(meta_data, colWidths=[250, 254])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 6))

    # ── 1. EXECUTIVE SUMMARY ───────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary & Problem Statement", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=5, spaceBefore=0))
    story.append(Paragraph(
        "<b>The Challenge:</b> Students and researchers grapple with dense, proprietary PDF materials (lecture notes, research papers, textbooks). Commercial cloud LLMs introduce three major barriers: (1) <i>Hallucinations</i> -- generating confident yet inaccurate statements ungrounded in course material; (2) <i>Data Privacy & Cost</i> -- requiring cloud uploads of confidential course documents and ongoing API fees; and (3) <i>Fragility</i> -- failing abruptly during free-tier rate limits (HTTP 429) or internet outages.",
        styles["BodyCustom"]
    ))
    story.append(Paragraph(
        "<b>The Solution:</b> The <b>AI Study Assistant</b> is a production-grade <b>Retrieval-Augmented Generation (RAG)</b> chatbot built from first principles in Python. It indexes PDFs into a local vector store and grounds LLM responses strictly in retrieved document passages with source citations (file name, page number, similarity score). The system features dual operational paradigms: a <b>100% air-gapped local mode</b> running local Ollama models (e.g. Qwen2.5-Coder:7B) with local embeddings, alongside a <b>zero-downtime multi-provider cloud fallback chain</b> with circuit-breaker rate-limit handling across Gemini, Groq, Cerebras, and OpenRouter.",
        styles["BodyCustom"]
    ))

    # ── 2. SYSTEM ARCHITECTURE & DATA FLOW ────────────────────────────────────
    story.append(Paragraph("2. System Architecture & Dual Pipeline Design", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=5, spaceBefore=0))
    story.append(Paragraph(
        "The application strictly decouples the <b>Document Indexing Pipeline</b> (offline ingest) from the <b>Query & Retrieval Pipeline</b> (runtime inference):",
        styles["BodyCustom"]
    ))

    pipeline_steps = [
        ("Step", "Component", "Implementation Details", "Output / Artifact"),
        ("1. Ingestion", "loader.py (pypdf)", "Page-by-page text extraction with UTF-8 normalization, empty page skipping, and scanned PDF detection.", "Structured dict list: file_name, page_num, raw text."),
        ("2. Chunking", "chunker.py", "Sliding-window segmentation (~800 characters, 150-char overlap) preserving semantic continuity.", "Fine-grained text chunks tagged with source metadata."),
        ("3. Embedding", "embedder.py (sentence-transformers)", "Local 'all-MiniLM-L6-v2' model converts text into dense 384-dimensional vectors; normalized with L2 norm.", "L2-normalized float32 numpy vector array (N, 384)."),
        ("4. Indexing", "vector_store.py (FAISS)", "FAISS IndexFlatIP computes exact inner products; persists index.faiss and chunks.pkl locally.", "Persistent vector index on disk; no re-embedding needed."),
        ("5. Retrieval", "rag_pipeline.py", "User question embedded with same model; FAISS retrieves top-k (default k=4) most relevant chunks via cosine similarity.", "Top-k retrieved passages with cosine scores & citations."),
        ("6. Generation", "fallback_manager.py", "Constructs grounded prompt ('Answer ONLY from context'); routes to Ollama or cloud providers with circuit breaker.", "Grounded, hallucination-resistant response with citations.")
    ]

    table_data = []
    for i, row in enumerate(pipeline_steps):
        if i == 0:
            table_data.append([Paragraph(cell, styles["TableHead"]) for cell in row])
        else:
            table_data.append([
                Paragraph(f"<b>{row[0]}</b>", styles["TableCellBold"]),
                Paragraph(row[1], styles["TableCellBold"]),
                Paragraph(row[2], styles["TableCell"]),
                Paragraph(row[3], styles["TableCell"])
            ])

    t_pipe = Table(table_data, colWidths=[55, 95, 220, 134])
    t_pipe.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_pipe)

    # ── 3. TECH STACK DECISIONS: WHAT WAS USED & WHY ───────────────────────────
    story.append(Paragraph("3. Tech Stack Decisions: What Was Used, Why, & Trade-Offs", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=5, spaceBefore=0))

    decisions = [
        ("Technology", "Role in System", "Why Chosen (Advantages)", "Alternative Rejected & Reason"),
        ("pypdf", "PDF text extraction", "Lightweight, pure Python, zero external binary/C dependencies, cross-platform reliable page extraction.", "PyMuPDF (heavy binaries, wheel installation issues on Windows student setups)."),
        ("sentence-transformers\n(all-MiniLM-L6-v2)", "Dense Vector Embedder", "Runs 100% locally on CPU or low-end GPU (4 GB VRAM). 384 dimensions, ~90 MB download, zero API latency, zero privacy leak.", "OpenAI text-embedding-3-small (requires cloud API key, billing, leaks notes online)."),
        ("FAISS\n(IndexFlatIP)", "Vector Store & Similarity Search", "Exact cosine similarity via L2-normalized inner product; deterministic; zero hyperparameter tuning; ultra-fast (<2ms for 10k vectors).", "Pinecone/ChromaDB (Pinecone requires remote cloud; Chroma adds heavy SQLite/DuckDB dependencies; HNSW approximation unneeded at this scale)."),
        ("Ollama\n(Qwen2.5-Coder:7B)", "Primary Local LLM Provider", "High reasoning quality, excellent prompt compliance, runs locally on 16 GB RAM / 4 GB VRAM GPU, zero internet needed.", "Llama-3-70B (too heavy for consumer hardware); GPT-4o (remote cloud dependency, subscription required)."),
        ("Multi-Provider Fallback\n(Gemini, Groq, Cerebras)", "Cloud Fallback LLM Chain", "Zero downtime. If one free tier hits 429 rate limit, automatically switches to next provider with circuit-breaker cooldown.", "Single-provider client (breaks immediately when free quota is exhausted during study sessions)."),
        ("Streamlit", "Reactive Frontend UI", "Fast Python-native UI; clean chat interface; session-state handles in-memory API keys safely without disk persistence.", "React/FastAPI (over-engineering for single-user study assistant; adds unnecessary frontend build tooling)."),
        ("Pytest & Unittest Mock", "Testing & Verification", "Full test coverage (67 tests) using synthetic embeddings and mocked HTTP requests; verifies pipeline without GPU or API keys.", "Manual browser testing (unreliable, non-reproducible, lacks regression safety).")
    ]

    t_dec_data = []
    for i, row in enumerate(decisions):
        if i == 0:
            t_dec_data.append([Paragraph(cell, styles["TableHead"]) for cell in row])
        else:
            t_dec_data.append([
                Paragraph(f"<b>{row[0]}</b>", styles["TableCellBold"]),
                Paragraph(row[1], styles["TableCell"]),
                Paragraph(row[2], styles["TableCell"]),
                Paragraph(row[3], styles["TableCell"])
            ])

    t_dec = Table(t_dec_data, colWidths=[90, 85, 165, 164])
    t_dec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_dec)

    # ── 4. MATHEMATICAL & ALGORITHMIC FOUNDATIONS ──────────────────────────────
    story.append(Paragraph("4. Mathematical & Algorithmic Foundations", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=5, spaceBefore=0))

    story.append(Paragraph("<b>A. Sliding-Window Text Chunking:</b>", styles["SubSectionHeader"]))
    story.append(Paragraph(
        "Splitting text by naive character slicing creates abrupt cuts through sentences and formulas. The sliding-window algorithm uses window size <i>W = 800</i> characters and step size <i>S = W - O = 650</i> characters (overlap <i>O = 150</i>). This guarantees that every sentence boundary has continuity in at least one chunk, preventing lost references when embedding.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("<b>B. Vector Normalization & Cosine Similarity in FAISS:</b>", styles["SubSectionHeader"]))
    story.append(Paragraph(
        "Cosine similarity between query vector <b>q</b> and document vector <b>d</b> is defined as: "
        "<b>cos(theta) = (q * d) / (||q|| * ||d||)</b>. By pre-normalizing all vectors to unit length using L2 normalization (<b>||v|| = 1</b>) during embedding, the denominator reduces to 1. Thus, cosine similarity becomes a pure <b>inner product: cos(theta) = q * d</b>. This allows <b>FAISS IndexFlatIP</b> to execute exact cosine similarity searches at hardware speeds using BLAS dot products without expensive trigonometric calculations.",
        styles["BodyCustom"]
    ))

    story.append(Paragraph("<b>C. Circuit Breaker & Rate-Limit Cooldown:</b>", styles["SubSectionHeader"]))
    story.append(Paragraph(
        "When an LLM provider returns HTTP 429 (Too Many Requests), naive loops will repeatedly hammer the rate-limited endpoint. <b>FallbackManager</b> captures the 429 and records a cooldown expiration: <i>T_cooldown = time.monotonic() + 60s</i>. Subsequent calls instantly bypass cooled-down providers without latency. If all providers fail, the system activates <b>Retrieval-Only Mode</b>, displaying the top-k passages directly so the user is never blocked.",
        styles["BodyCustom"]
    ))

    # ── 5. SECURITY, PRIVACY & PRODUCTION RESILIENCY ───────────────────────────
    story.append(Paragraph("5. Security, Privacy & Production Resiliency", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=5, spaceBefore=0))

    story.append(Paragraph("• <b>100% Air-Gapped Local Mode:</b> When the Offline Mode toggle is enabled, zero bytes leave the machine. All vector calculations occur in memory, and completions run through local Ollama.", styles["BulletCustom"]))
    story.append(Paragraph("• <b>In-Memory Key Injection (BYOK):</b> API keys entered via the Streamlit UI are stored exclusively in temporary session memory (<code>st.session_state</code>). They are never written to disk, never logged, and destroyed upon tab closure.", styles["BulletCustom"]))
    story.append(Paragraph("• <b>Strict Credential Isolation:</b> The <code>.env</code> file is enforced in <code>.gitignore</code>. A comprehensive template <code>.env.example</code> provides documentation without risk of credential leakage.", styles["BulletCustom"]))
    story.append(Paragraph("• <b>Persistent Vector Cache:</b> Generated FAISS indexes and metadata pickles are serialized to <code>data/</code>, allowing instant reload across app restarts without re-embedding.", styles["BulletCustom"]))

    # ── 6. COMPREHENSIVE INTERVIEW PREPARATION GUIDE ───────────────────────────
    story.append(Paragraph("6. Technical Interview Guide: Questions & Model Answers", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=5, spaceBefore=0))

    interview_qa = [
        (
            "Q1: Why did you build a RAG pipeline instead of fine-tuning an open-source model?",
            "Fine-tuning modifies model weights to alter style or domain language, but it cannot guarantee grounded facts, cannot cite exact page numbers, requires expensive GPU compute, and cannot be updated dynamically when notes change. RAG decouples knowledge storage (FAISS index) from reasoning (LLM). When new notes are uploaded, they are indexed in seconds with zero training cost, and answers can always be verified against source citations."
        ),
        (
            "Q2: How does FAISS IndexFlatIP work and why did you choose it over approximate indexes like IVFFlat or HNSW?",
            "IndexFlatIP performs exhaustive, exact inner-product search across all stored vectors. Because we L2-normalize all vectors upon embedding, the inner product is mathematically identical to cosine similarity. Approximate nearest neighbor (ANN) indexes like IVFFlat or HNSW trade accuracy for speed by clustering vectors, which introduces false negatives and requires hyperparameter tuning (nlist, nprobe). For document collections under 100,000 chunks, exact search takes under 2ms, so exact precision is vastly preferable."
        ),
        (
            "Q3: How do you prevent LLM hallucinations when answering questions?",
            "Hallucination mitigation relies on three layers: (1) Prompt Constraints: The system prompt instructs the model: 'Answer using ONLY the context passages provided. If not in the context, respond with: I couldn't find this in your documents.' (2) Low Temperature: Setting temperature to 0.2 prioritizes determinism over creativity. (3) Verifiable Source Attribution: Every answer exposes the file name, page number, and similarity score so the student can verify the snippet immediately."
        ),
        (
            "Q4: How does your fallback mechanism handle provider outages and rate limits?",
            "The FallbackManager implements an ordered provider chain with an automated circuit breaker. When a provider returns an HTTP 429 status code, the manager registers a 60-second cooldown via time.monotonic() and seamlessly falls through to the next provider (e.g. Gemini -> Groq -> Cerebras -> OpenRouter -> Ollama). Subsequent calls skip cooled-down providers with zero latency until the timer expires. If all providers are unavailable, the system degrades gracefully into 'retrieval-only' mode."
        ),
        (
            "Q5: How did you ensure testability across the entire pipeline without depending on cloud APIs?",
            "We built a 67-test pytest suite using dependency injection and unittest.mock. The vector store is validated using deterministic synthetic vectors; the fallback manager tests mock HTTP status codes (200, 401, 429, 500) to verify failover logic; and pypdf loaders are validated with dynamically generated blank and text PDFs in temporary directories."
        )
    ]

    for q, a in interview_qa:
        story.append(Paragraph(f"<b>{q}</b>", styles["BodyBold"]))
        story.append(Paragraph(a, styles["BodyCustom"]))
        story.append(Spacer(1, 3))

    # ── 7. RESUME BULLET POINTS ───────────────────────────────────────────────
    story.append(Paragraph("7. Resume Bullets (Action Verb + Tech + Impact Framework)", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=5, spaceBefore=0))

    story.append(Paragraph(
        "• <b>Architected an end-to-end RAG AI Study Assistant</b> in Python (Streamlit, FAISS, sentence-transformers) enabling grounded semantic Q&A over multi-page PDF documents with verifiable source citations and page numbers.",
        styles["BulletCustom"]
    ))
    story.append(Paragraph(
        "• <b>Engineered a zero-downtime LLM fallback engine</b> supporting 5 providers (Gemini, Groq, Cerebras, OpenRouter, Ollama) with automated HTTP 429 circuit breaking, 60s cooldowns, and graceful retrieval-only degradation.",
        styles["BulletCustom"]
    ))
    story.append(Paragraph(
        "• <b>Implemented privacy-first air-gapped architecture</b> supporting quantized local models (Qwen2.5-Coder:7B / Llama 3.2:3B) on 4 GB VRAM consumer GPUs with local 384-dim embeddings and zero cloud data transmission.",
        styles["BulletCustom"]
    ))
    story.append(Paragraph(
        "• <b>Authored a robust 67-test automated test suite</b> (pytest) achieving 100% component validation across chunking, L2 vector normalization, FAISS persistence, and fault-tolerant provider failover.",
        styles["BulletCustom"]
    ))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] PDF successfully generated at: {output_filename}")


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent.parent / "AI_Study_Assistant_Project_Documentation.pdf"
    build_pdf(str(out_path))
