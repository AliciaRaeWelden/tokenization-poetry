"""
poem → color → meaning
what tokenization does to poetry, in pictures.

A focused demo built around four poems where the tokenizer reveals
something specific about how language models read text.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorsys
import hashlib
from collections import Counter

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="poem → color",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stApp { background: #0f0e0c; color: #e8e6df; }
  h1, h2, h3 {
    font-family: 'Iowan Old Style', 'Palatino', 'Georgia', serif !important;
    font-weight: 400 !important;
    letter-spacing: -0.01em;
  }
  h1 { font-size: 2.4rem !important; }
  h2 { font-size: 1.6rem !important; margin-top: 1.5rem !important; }
  h3 { font-size: 1.2rem !important; }
  .stTextArea textarea {
    font-family: 'Iowan Old Style', 'Palatino', 'Georgia', serif !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
    background: #1a1815 !important;
    color: #e8e6df !important;
  }
  .token-grid { display: flex; flex-wrap: wrap; gap: 2px; margin: 8px 0 16px; }
  .token-cell {
    min-width: 32px; height: 32px;
    display: inline-flex; align-items: center; justify-content: center;
    font-family: 'SF Mono', 'Menlo', monospace; font-size: 10px;
    padding: 0 6px; border-radius: 3px;
  }
  .meta { color: #8a8678; font-size: 13px; line-height: 1.5; }
  .commentary {
    font-family: 'Iowan Old Style', 'Palatino', 'Georgia', serif;
    font-size: 15px; line-height: 1.65; color: #c8c4b8;
    padding: 12px 16px; border-left: 2px solid #c47a3d;
    background: rgba(196, 122, 61, 0.04); margin: 12px 0;
  }
  .primer {
    font-family: 'Iowan Old Style', 'Palatino', 'Georgia', serif;
    font-size: 15px; line-height: 1.7; color: #c8c4b8;
  }
  .primer p { margin: 0 0 12px; }
  .primer strong { color: #e8e6df; font-weight: 500; }
  .primer code { background: #1a1815; padding: 1px 6px; border-radius: 3px; color: #c47a3d; font-size: 13px; }
  .legend {
    display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0;
    font-size: 12px; color: #8a8678; align-items: center;
  }
  .legend-cell {
    display: inline-flex; align-items: center; gap: 6px;
  }
  .legend-swatch {
    width: 18px; height: 18px; border-radius: 3px; display: inline-block;
  }
  .finding {
    background: #1a1815; border: 0.5px solid #3a3835;
    padding: 14px 18px; border-radius: 6px; margin: 12px 0;
  }
  .finding-title {
    font-family: 'Iowan Old Style', serif;
    font-size: 14px; color: #c47a3d; margin-bottom: 8px;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  hr { border-color: #2a2825; margin: 1.5rem 0 !important; }
  .stMarkdown code { background: #1a1815; padding: 2px 6px; border-radius: 3px; color: #c8c4b8; }
  /* tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; border-bottom: 0.5px solid #3a3835; }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #8a8678 !important;
    font-family: 'Iowan Old Style', serif !important; font-size: 15px !important;
    padding: 8px 14px !important;
  }
  .stTabs [aria-selected="true"] { color: #e8e6df !important; border-bottom: 2px solid #c47a3d !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# TOKENIZERS
# ============================================================
@st.cache_resource
def load_tiktoken_cl100k():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")

@st.cache_resource
def load_llama():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception:
        return AutoTokenizer.from_pretrained("t5-small")

@st.cache_resource
def load_bert():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_cl100k(text):
    enc = load_tiktoken_cl100k()
    ids = enc.encode(text)
    return [(enc.decode([i]), i) for i in ids]

def tokenize_llama(text):
    tok = load_llama()
    enc = tok(text, add_special_tokens=False)
    return [(tok.decode([i]), i) for i in enc["input_ids"]]

def tokenize_bert(text):
    tok = load_bert()
    enc = tok(text, add_special_tokens=False)
    return [(tok.decode([i]), i) for i in enc["input_ids"]]

def tokenize_char(text):
    return [(c, ord(c)) for c in text]


TOKENIZERS = {
    "gpt-4 (cl100k)": tokenize_cl100k,
    "llama (sentencepiece)": tokenize_llama,
    "bert (wordpiece)": tokenize_bert,
    "character-level": tokenize_char,
}

VOCAB_SIZES = {
    "gpt-4 (cl100k)": lambda: load_tiktoken_cl100k().n_vocab,
    "llama (sentencepiece)": lambda: load_llama().vocab_size,
    "bert (wordpiece)": lambda: load_bert().vocab_size,
    "character-level": lambda: 0x110000,
}


# ============================================================
# EMBEDDINGS + NEIGHBORS
# ============================================================
@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

COMMON_WORDS_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa-no-swears-medium.txt"

@st.cache_resource
def build_neighbor_corpus():
    """Encode a 6K-word common-English vocab for nearest-neighbor lookup."""
    import urllib.request
    import random
    vocab = None
    try:
        with urllib.request.urlopen(COMMON_WORDS_URL, timeout=10) as r:
            text = r.read().decode("utf-8")
        vocab = sorted({w.strip().lower() for w in text.splitlines()
                        if w.strip().isalpha() and 3 <= len(w.strip()) <= 12})
    except Exception:
        # Fallback: a small hand-curated list
        vocab = ["red", "blue", "green", "color", "white", "black", "yellow",
                 "wheel", "wheels", "wagon", "tire", "bicycle", "cart",
                 "valley", "village", "cottage", "manor", "woods", "meadow",
                 "rain", "water", "weather", "cloudy", "moisture", "flood",
                 "chicken", "chickens", "poultry", "bird", "birds",
                 "cooked", "salad", "polished", "spice", "shiny", "wet",
                 "spring", "summer", "autumn", "winter", "season", "snow",
                 "person", "people", "human", "anyone", "someone", "nobody",
                 "live", "lived", "alive", "living", "death", "die",
                 "sang", "song", "singing", "singer", "music",
                 "danced", "dance", "dancing", "movement",
                 "happened", "worked", "tried", "did", "done"]
    embs = load_embedder().encode(vocab, show_progress_bar=False,
                                   batch_size=128, normalize_embeddings=True)
    return vocab, embs


def find_neighbors(text, k=5):
    text = text.strip().lower()
    if not text or len(text) < 1:
        return []
    vocab, embs = build_neighbor_corpus()
    q = load_embedder().encode([text], normalize_embeddings=True)[0]
    sims = embs @ q
    top = np.argsort(-sims)[:k + 2]
    out = []
    for i in top:
        if vocab[i] == text:
            continue
        out.append((vocab[i], float(sims[i])))
        if len(out) >= k:
            break
    return out


# ============================================================
# COLOR
# ============================================================
@st.cache_resource
def fit_color_pca():
    """Fit a PCA on anchor sentences so embeddings can be projected to RGB."""
    from sklearn.decomposition import PCA
    anchors = [
        "red orange fire warm hot sun blood",
        "green grass forest leaf nature plant tree",
        "blue ocean sky cold water deep night ice",
        "person body human face hand voice",
        "thought mind idea spirit dream meaning",
        "object thing stone metal wood",
    ]
    embs = load_embedder().encode(anchors, normalize_embeddings=True)
    pca = PCA(n_components=3)
    pca.fit(embs)
    return pca

def color_for_token(token_text, mode="embedding"):
    """Return a CSS rgb() string for a token."""
    if mode == "embedding":
        try:
            pca = fit_color_pca()
            e = load_embedder().encode([token_text], normalize_embeddings=True)
            proj = pca.transform(e)[0]
            proj = np.clip((proj + 0.5) / 1.0, 0, 1)
            r, g, b = (proj * 255).astype(int)
            return f"rgb({r},{g},{b})"
        except Exception:
            return "rgb(120,120,120)"
    else:  # hue mode by id (passed token_text is actually id here)
        h = (int(token_text) % 360) / 360.0
        r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.65)
        return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def render_token_strip(pieces, color_mode="embedding"):
    cells = []
    for text, tid in pieces:
        if color_mode == "embedding":
            color = color_for_token(text if text.strip() else " ", "embedding")
        else:
            color = color_for_token(tid, "hue")
        rgb = [int(x) for x in color[4:-1].split(",")]
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        fg = "#000" if lum > 140 else "#fff"
        display = text.replace("\n", "⏎").replace(" ", "␣")[:8]
        # tooltip is escaped basic
        title = f'"{text}" → id {tid}'.replace('"', "&quot;")
        cells.append(
            f'<span class="token-cell" style="background:{color};color:{fg};" title="{title}">{display}</span>'
        )
    return f'<div class="token-grid">{"".join(cells)}</div>'


# ============================================================
# NUMBER LINE
# ============================================================
def plot_number_line(ids, vocab_size, label):
    if len(ids) == 0:
        return None
    ids = np.asarray(ids)
    fig, (ax_full, ax_zoom) = plt.subplots(
        2, 1, figsize=(11, 2.6), facecolor="#0f0e0c",
        gridspec_kw={"height_ratios": [1, 1], "hspace": 1.0},
    )
    for ax in (ax_full, ax_zoom):
        ax.set_facecolor("#1a1815")
        for s in ax.spines.values():
            s.set_color("#3a3835")
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(colors="#8a8678", labelsize=9)
        ax.set_yticks([])

    ax_full.set_xlim(0, vocab_size)
    ax_full.set_ylim(0, 1)
    ax_full.scatter(ids, [0.5] * len(ids), marker="|", s=400, linewidths=1.2,
                    color="#c47a3d", alpha=0.85)
    ax_full.set_title(
        f"full vocab: 0 → {vocab_size:,}    "
        f"({len(ids)} tokens, min={int(ids.min()):,}  max={int(ids.max()):,})",
        color="#e8e6df", fontsize=10, loc="left", pad=8,
    )
    ax_full.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    span = ids.max() - ids.min()
    pad = max(span * 0.05, 50)
    lo, hi = max(0, ids.min() - pad), min(vocab_size, ids.max() + pad)
    ax_zoom.set_xlim(lo, hi)
    ax_zoom.set_ylim(0, 1)
    ax_zoom.scatter(ids, [0.5] * len(ids), marker="|", s=400, linewidths=1.2,
                    color="#7aa8c8", alpha=0.85)
    ax_zoom.set_title(
        f"zoomed: {int(lo):,} → {int(hi):,}    "
        f"(span = {int(span):,} = {span/vocab_size*100:.2f}% of vocab)",
        color="#e8e6df", fontsize=10, loc="left", pad=8,
    )
    ax_zoom.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_full.axvspan(lo, hi, color="#7aa8c8", alpha=0.12)
    plt.tight_layout()
    return fig


# ============================================================
# NEIGHBOR TABLE
# ============================================================
def render_neighbor_table(pieces, color_mode="embedding"):
    rows = []
    seen = set()
    for text, tid in pieces:
        key = text.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            neighbors = find_neighbors(text, k=5)
            n_str = "  ·  ".join(f"{w} ({s:.2f})" for w, s in neighbors) or "(no neighbors)"
        except Exception as e:
            n_str = f"(error: {e})"
        if color_mode == "embedding":
            color = color_for_token(text, "embedding")
        else:
            color = color_for_token(tid, "hue")
        rgb = [int(x) for x in color[4:-1].split(",")]
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        fg = "#000" if lum > 140 else "#fff"
        display = text.replace("\n", "⏎").replace(" ", "␣")
        rows.append(
            f'<tr>'
            f'<td style="padding:6px 12px 6px 0;"><span class="token-cell" '
            f'style="background:{color};color:{fg};min-width:60px;">{display}</span></td>'
            f'<td style="padding:6px 12px;color:#8a8678;font-family:SF Mono,Menlo,monospace;font-size:11px;">{tid}</td>'
            f'<td style="padding:6px 0;color:#c8c4b8;font-size:13px;">{n_str}</td>'
            f'</tr>'
        )
    return (
        '<table style="border-collapse:collapse;width:100%;">'
        '<thead><tr style="border-bottom:0.5px solid #3a3835;">'
        '<th style="text-align:left;padding:6px 12px 6px 0;color:#8a8678;font-weight:400;font-size:12px;">token</th>'
        '<th style="text-align:left;padding:6px 12px;color:#8a8678;font-weight:400;font-size:12px;">id</th>'
        '<th style="text-align:left;padding:6px 0;color:#8a8678;font-weight:400;font-size:12px;">5 nearest words (cosine sim)</th>'
        '</tr></thead><tbody>'
        + "".join(rows) +
        '</tbody></table>'
    )


# ============================================================
# ANALYSIS PANEL — used for both presets and custom input
# ============================================================
def analyze(text, tokenizer_name, show_neighbors=False, color_mode="embedding"):
    """Render the standard analysis: token strip + number line + (optional) neighbors."""
    fn = TOKENIZERS[tokenizer_name]
    pieces = fn(text)
    ids = [p[1] for p in pieces]

    st.markdown(
        f'<p class="meta"><strong style="color:#c8c4b8;">{tokenizer_name}</strong> '
        f'· {len(pieces)} tokens · color mode: <code>{color_mode}</code></p>',
        unsafe_allow_html=True,
    )
    st.markdown(render_token_strip(pieces, color_mode), unsafe_allow_html=True)

    with st.expander("token ids on the number line"):
        try:
            vocab_size = VOCAB_SIZES[tokenizer_name]()
            fig = plot_number_line(ids, vocab_size, tokenizer_name)
            if fig:
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"could not load vocab size: {e}")

    if show_neighbors:
        with st.expander("what each token means to the model (semantic neighbors)", expanded=True):
            with st.spinner("computing neighbors (slow first time, cached after)…"):
                build_neighbor_corpus()  # warm cache
            st.markdown(render_neighbor_table(pieces, color_mode), unsafe_allow_html=True)

    return pieces


# ============================================================
# POEMS + COMMENTARY
# ============================================================
POEMS = {
    "Williams — The Red Wheelbarrow": {
        "text": "so much depends\nupon\n\na red wheel\nbarrow\n\nglazed with rain\nwater\n\nbeside the white\nchickens",
        "tokenizer": "llama (sentencepiece)",
        "headline": "the model cannot see the wheelbarrow",
        "intro": (
            "Williams's 1923 poem hinges on a single image: a red wheelbarrow, wet with rain, in a "
            "farmyard with white chickens. The line break between 'wheel' and 'barrow' is the whole point — "
            "it splits the compound noun, makes you see the parts, then fuses them back together in the mind's eye. "
            "What happens when a tokenizer reads it?"
        ),
        "findings": [
            ("the compound dissolves",
             "Under any subword tokenizer, 'wheelbarrow' isn't in the vocabulary as a single token. "
             "The model sees 'wheel' and 'barrow' separately. 'Wheel' is fine — its semantic neighbors are "
             "'wheels, tires, bicycle, wagon, rover.' The model knows wheels go on vehicles. "
             "But 'barrow' on its own? Its neighbors are <strong>valley, woods, village, cottage, manor</strong>. "
             "The model reads it as the British topographical 'barrow' — a burial mound, a feature of the English countryside. "
             "It has no idea this is the back half of a gardening tool."),
            ("'glazed' becomes food",
             "Williams's 'glazed with rain water' means the wet sheen on red paint. "
             "The model's neighbors for 'glazed': <strong>salad, polished, spice, cooked, cuisine</strong>. "
             "Only one of those (polished) captures the intended meaning. The dominant reading is culinary."),
            ("what survives, what doesn't",
             "Of the poem's 16 words, the model gets ~14 right in isolation. "
             "But the two it gets wrong — 'barrow' and 'glazed' — are exactly the words that make this an image rather than a list. "
             "Without them, you're left with: some color words, some weather words, some chickens, a wheel. "
             "The poem becomes a pastoral inventory. The wheelbarrow disappears."),
        ],
        "takeaway": (
            "This is the compositionality problem in miniature. Meaning isn't just the average of word meanings; "
            "it lives in their relationships. Tokenization + embedding flattens compositional meaning into "
            "bag-of-concepts, and the things most easily lost are the things poetry depends on most: "
            "compound images, deliberate enjambment, the way two words touching make a third meaning."
        ),
    },
    "cummings — anyone lived in a pretty how town": {
        "text": "anyone lived in a pretty how town\n(with up so floating many bells down)\nspring summer autumn winter\nhe sang his didn't he danced his did",
        "tokenizer": "gpt-4 (cl100k)",
        "headline": "the seasons cluster, the grammar doesn't",
        "intro": (
            "cummings does grammatical violence on purpose: 'anyone' becomes a name, 'how' an adjective, "
            "'didn't' and 'did' get used as nouns. The poem only works if you let language come unglued. "
            "What does an embedding model — trained on conventional grammar — do with it?"
        ),
        "findings": [
            ("the seasons form a tight cluster",
             "Look at the colors: 'spring,' 'summer,' 'autumn,' 'winter' all land in the warm-red region. "
             "Their semantic neighbors confirm it — each season's nearest words are the other three seasons, "
             "plus 'season,' 'seasonal,' 'december.' The model has learned 'season-ness' as a coherent region "
             "in embedding space, and you can literally see it as a chromatic family in the strip above."),
            ("function words become mush",
             "'anyone, lived, in, a, pretty, how, town' — most of these end up muted browns and grays. "
             "Function words and grammar scaffolding sit near the centroid of embedding space because they have "
             "weak, contextual meaning on their own. The poem's <em>syntax</em> is brown; only the content words have color."),
            ("the model misses cummings's trick",
             "cummings's punchline is using 'did' and 'didn't' as nouns ('he sang his didn't he danced his did'). "
             "But the model sees 'did → happened, worked, tried, voted, showed' (regular past-tense verbs) and "
             "'didn't → supposed, already, happened' (regular auxiliary). The grammatical inversion is "
             "<strong>invisible to the embedding</strong> — the words are averaged over all their conventional uses, "
             "and cummings's specific weird use is washed out."),
        ],
        "takeaway": (
            "Embeddings know about semantic clusters (seasons, colors, kinship) but they don't know about "
            "grammatical context-shifts within a single poem. The model sees the season-cycle clearly and the "
            "syntactic violence not at all. cummings would have found this funny: the machine that reads everything "
            "fails specifically on the moves he made his entire career out of."
        ),
    },
    "Bashō — old pond": {
        "text": "An old silent pond...\nA frog jumps into the pond,\nsplash! Silence again.",
        "tokenizer": "gpt-4 (cl100k)",
        "headline": "compare across all four schemes",
        "intro": (
            "This haiku is short enough to see all four tokenizers at once without scrolling. "
            "Watch how the same 16 words become wildly different sequences depending on the scheme."
        ),
        "findings": [
            ("each tokenizer sees a different poem",
             "Character-level produces ~80 tokens — every letter is a token, every Unicode codepoint is an ID. "
             "The colors form gentle gradients because adjacent characters have adjacent IDs. "
             "BERT WordPiece produces ~16. LLaMA SentencePiece ~20. GPT-4 cl100k ~12. "
             "The same line of poetry has four different lengths depending on who's counting."),
            ("the number-line view shows the philosophy",
             "Open the number-line panel for each tokenizer and notice: BPE schemes cluster tokens at the low end "
             "(common words = small IDs by frequency rank). Character-level clusters around ASCII (32-127). "
             "<strong>The shape of the distribution is the philosophy of the tokenizer</strong> — what it thinks "
             "is worth encoding in the integer itself."),
            ("'splash' is interesting",
             "In Bashō's original Japanese ('Furuike ya / kawazu tobikomu / mizu no oto'), the splash is 'mizu no oto' — "
             "literally 'water sound.' English translators chose the onomatopoeia 'splash!' which most BPE tokenizers "
             "have as a single token. Try replacing 'splash' with 'plop' or 'kerplunk' and watch the token count change. "
             "Onomatopoeia is a stress test for tokenizer vocabularies."),
        ],
        "takeaway": (
            "Tokenization isn't a neutral preprocessing step. It's an opinion about what units of language "
            "are worth tracking. Different opinions produce wildly different representations of the same text — "
            "and a short poem makes the differences impossible to miss."
        ),
    },
    "Dickinson — Because I could not stop for Death": {
        "text": "Because I could not stop for Death,\nHe kindly stopped for me;\nThe carriage held but just ourselves\nAnd Immortality.",
        "tokenizer": "gpt-4 (cl100k)",
        "headline": "where the demo started",
        "intro": (
            "The first poem we tokenized. A useful baseline: standard 19th-century English, mostly common words, "
            "a handful of capitalized abstractions ('Death,' 'Immortality'). Look at how those abstractions are handled."
        ),
        "findings": [
            ("'Immortality' breaks into pieces",
             "Even cl100k, with its 100K-token vocabulary, doesn't have 'Immortality' as a single token. "
             "It splits into 'Imm', 'ort', 'ality' (or similar). Long, abstract, capitalized words almost always fragment. "
             "Compare to 'Death' which usually stays whole — it's shorter and far more frequent."),
            ("punctuation costs tokens",
             "The semicolon, comma, and period each get their own token. So does the trailing newline. "
             "On a 4-line poem this is ~10% of the token count. For poetry — which uses punctuation and line breaks "
             "deliberately — this matters: every formal choice is also a tokenization cost."),
            ("the 'ourselves' surprise",
             "'Ourselves' is a single token in cl100k. So is 'kindly.' But 'Immortality' isn't. "
             "Tokenizer vocabularies are built from training data, and what they keep whole reveals what was common "
             "in that data — internet English, mostly. Reflexive pronouns and adverbs: yes. Latinate abstractions: less so."),
        ],
        "takeaway": (
            "Even on 'normal' English, the tokenizer makes choices about what's worth a single integer. "
            "The choices reflect the corpus they were trained on, not the formal qualities of literary language. "
            "Poems that use uncommon vocabulary or formal diction get fragmented more aggressively than colloquial ones."
        ),
    },
    "Lemay — In Lieu of Flowers": {
        "text": (
            "Although I love flowers very much, I won't see them when I'm gone. "
            "So in lieu of flowers: Buy a book of poetry written by someone still alive, "
            "sit outside with a cup of tea, a glass of wine, and read it out loud, by yourself or to someone, or silently. "
            "Spend some time with a single flower. A rose maybe. Smell it, touch the petals. Really look at it. "
            "Drink a nice bottle of wine with someone you love. Or, Champagne. "
            "And think of what John Maynard Keynes said, \"My only regret in life is that I did not drink more Champagne.\" "
            "Or what Dom Perignon said when he first tasted the stuff: \"Come quickly! I am tasting stars!\" "
            "Take out a paint set and lay down some colours. "
            "Watch birds. Common sparrows are fine. Pigeons, too. Geese are nice. Robins. "
            "In lieu of flowers, walk in the trees and watch the light fall into it. Eat an apple, a really nice big one. "
            "I hope it's crisp. Have a long soak in the bathtub with candles, maybe some rose petals. "
            "Sit on the front stoop and watch the clouds. Have a dish of strawberry ice cream in my name. "
            "If it's winter, have a cup of hot chocolate outside for me. If it's summer, a big glass of ice water. "
            "If it's autumn, collect some leaves and press them in a book you love. "
            "Sit and look out a window and write down what you see. Write some other things down. "
            "In lieu of flowers, I would wish for you to flower. I would wish for you to blossom, to open, to be beautiful."
        ),
        "tokenizer": "bert (wordpiece)",
        "headline": "the model lights up exactly what the poem bequeaths",
        "intro": (
            "Shawna Lemay's poem-essay, from <em>The Flower Can Always Be Changing</em> (2018). "
            "She wrote it after reading a friend's father's obituary on Facebook — he'd asked mourners "
            "to take a loved one out for lunch instead of sending flowers. "
            "Lemay expanded the request into her own: what she'd want people to do when she's gone. "
            "It's a funeral poem disguised as a list of small pleasures. "
            "Watch what the embedding model lights up — and what it can't see at all."
        ),
        "findings": [
            ("the colored cells are the bequest",
             "Run the embedding-PCA color mode and look at which cells glow brightest: "
             "<strong>flowers, trees, ice, summer, winter, autumn, leaves, blossom</strong>. "
             "These are the things Lemay is leaving to her readers — the sensory inheritances she can't take with her. "
             "The function words connecting them (with, of, for, in, to) sit gray and muted. "
             "<em>The grammar of giving is the color of mud; the things being given are the bright pixels.</em>"),
            ("the seasons cluster again, but the cycle means something different here",
             "Just like cummings, the seasons (summer, winter, autumn) form a tight chromatic cluster. "
             "But here the cycle isn't decoration — it's a structural device: 'if it's winter, have hot chocolate; "
             "if it's summer, ice water; if it's autumn, collect leaves.' Lemay uses the year's full cycle to say "
             "<em>whenever you remember me, here's what to do.</em> The model sees the season-cluster the same way "
             "it saw it in cummings, but the poem is using that cluster to encode the passage of time-without-her."),
            ("hope, wish, think — purple, and only at the end",
             "Look at the placement of the few mental-state words in the poem. "
             "<strong>think</strong> appears once, mid-poem (about Keynes). "
             "<strong>hope</strong> appears once ('I hope it's crisp'). "
             "<strong>wish</strong> appears twice — both in the final lines: "
             "'I would wish for you to flower. I would wish for you to blossom.' "
             "The embedding puts these in the purple-violet family, visually distinct from everything else. "
             "<em>The poem withholds its interiority until the final lines, then floods.</em> "
             "The visualization makes the emotional architecture visible."),
            ("the famous quotes get destroyed",
             "Lemay quotes John Maynard Keynes and Dom Pérignon. Watch what BERT does: "
             "<strong>'Champagne'</strong> splits into <code>champagn</code> + <code>##e</code>. "
             "<strong>'Dom Perignon'</strong> splits into <code>dom</code> + <code>per</code> + <code>##ignon</code> — "
             "three meaningless fragments. The most famous champagne in the world becomes nothing the model can recognize. "
             "<strong>'John Maynard Keynes'</strong> stays as separate proper-noun tokens, all muddy. "
             "The cultural references — the things Lemay assumes her reader will recognize — "
             "are exactly the things the tokenizer cannot."),
            ("the model can't see what kind of poem this is",
             "Nothing in the embedding tells the model this is a funeral poem. It doesn't know 'someone still alive' "
             "is a tell. It doesn't know the speaker is dead, addressing the living. It doesn't know the title means "
             "<em>instead of the flowers you'd send to my funeral.</em> All of that lives in human framing — the genre, "
             "the cultural convention, the specific context of an obituary. "
             "The embedding makes the bright-pixel pattern visible to us, and we supply the meaning. "
             "<strong>The gap between what the model can see (semantic categories) and what we can see "
             "(genre, mortality, address) might be the deepest thing this demo accidentally reveals.</strong>"),
        ],
        "takeaway": (
            "This poem makes the strongest case for what tokenization + embedding actually does: "
            "it reveals semantic structure (color clusters, mental-state words, season cycles) "
            "while remaining entirely blind to genre, intention, and address. "
            "Lemay's poem is a gift from a dying woman to the living. The model sees a list of nouns about flowers and weather. "
            "The bright pixels are real — they correspond to real semantic regions in language. "
            "But the meaning of the poem is not in the pixels. It's in what we bring to them: "
            "the knowledge that someone is saying goodbye. That part is ours, not the model's, "
            "and probably always will be."
        ),
    },
}


# ============================================================
# UI
# ============================================================
st.title("poem → color")
st.markdown(
    '<p class="meta" style="margin-bottom: 1.2rem;">'
    'what tokenization does to poetry, in pictures.'
    '</p>',
    unsafe_allow_html=True,
)

# 30-second primer — visible by default
st.markdown(
    '<div class="primer" style="max-width: 780px;">'
    '<p>Before a language model can read a poem, it has to break the text into pieces called <strong>tokens</strong> '
    'and assign each piece a number. Sometimes a token is a whole word (<code>flowers</code>). '
    'Often it\'s a fragment (<code>champagn</code> + <code>##e</code>). '
    'These choices aren\'t neutral — they shape what the model can actually see.</p>'
    '<p>This demo turns each token into a colored cell. The color comes from the token\'s '
    '<strong>embedding</strong> — a high-dimensional vector the model uses to encode meaning. '
    'We squash that vector down to three numbers and call them red, green, and blue. '
    'The result: <strong>tokens with similar meanings get similar colors</strong>. '
    'Words about warmth and life cluster in one color region; words about cold or water in another; '
    'meaningless subword fragments end up muddy gray.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Visual legend
st.markdown(
    '<div class="legend">'
    '<span style="color:#c8c4b8; font-weight:500;">how to read the colored strips:</span>'
    '<span class="legend-cell"><span class="legend-swatch" style="background:#d4543a;"></span>warm = fire / life / red</span>'
    '<span class="legend-cell"><span class="legend-swatch" style="background:#7aa86a;"></span>green = nature / objects</span>'
    '<span class="legend-cell"><span class="legend-swatch" style="background:#5a7a9a;"></span>blue = water / cold / abstract</span>'
    '<span class="legend-cell"><span class="legend-swatch" style="background:#7a6a78;"></span>muddy = function words / fragments</span>'
    '</div>',
    unsafe_allow_html=True,
)

# Optional deeper dive — collapsed by default
with st.expander("a slightly longer explanation"):
    st.markdown(
        '<div class="primer">'

        '<p><strong>Why tokenization isn\'t just splitting on spaces.</strong> '
        'You might think a tokenizer would just cut text at the whitespace and call each word a token. '
        'Some do (we include one called <code>character-level</code> for comparison). '
        'But most modern language models — GPT, Claude, BERT, LLaMA — use <strong>subword tokenization</strong>. '
        'They start with a fixed vocabulary of, say, 100,000 common pieces (whole words, common roots, frequent fragments) '
        'and any text gets greedily matched against it. Common words stay whole; uncommon words get chopped into '
        'recognizable pieces. <code>wheelbarrow</code> isn\'t in the vocab, so it becomes <code>wheel</code> + <code>barrow</code>. '
        '<code>Immortality</code> becomes <code>Imm</code> + <code>ort</code> + <code>ality</code>. '
        'These chopping decisions are made once, when the tokenizer is trained, and they\'re frozen forever after.</p>'

        '<p><strong>Why each token gets a number.</strong> '
        'After the text is broken into tokens, each token is replaced by its position in the vocabulary — '
        'an integer ID. The word <code>the</code> might be ID 264. <code>flowers</code> might be 81154. '
        'Models do all their math on these integers (and the vectors associated with them), never on the original text. '
        'You can see these IDs in the table under each poem.</p>'

        '<p><strong>What an embedding is.</strong> '
        'Each token ID points to a row in a giant lookup table the model has learned. '
        'That row is a list of ~384 numbers (in the model we\'re using, MiniLM-L6-v2). '
        'These numbers — the <strong>embedding</strong> — encode what the model "knows" about the token: '
        'which other tokens it tends to appear near, what topics it shows up in, what grammatical role it plays. '
        'Two tokens with similar embeddings are tokens the model treats as semantically related. '
        'When you see <code>summer</code>, <code>winter</code>, <code>autumn</code>, <code>spring</code> all glow the same warm color, '
        'that\'s because their 384-dimensional embeddings are nearly identical — the model has learned "season-ness" as a region.</p>'

        '<p><strong>How we\'re turning embeddings into colors.</strong> '
        'A 384-dimensional vector can\'t be displayed directly. So we use a technique called PCA (principal component analysis) '
        'to project each embedding down to 3 dimensions, fitted on a small set of anchor sentences about warmth, nature, and water. '
        'Those 3 numbers become RGB. It\'s a lossy compression, but it preserves the overall structure: '
        'tokens close in meaning end up close in color.</p>'

        '<p><strong>What to look for in each poem.</strong> '
        'Bright, distinct colors usually mean strong semantic content (concrete nouns, vivid verbs). '
        'Muddy browns and grays usually mean function words (the, of, with) or meaningless subword fragments. '
        'When two unrelated words share a color, the model considers them semantically related. '
        'When a word you expect to be vivid comes out muddy, it\'s probably been fragmented and the model is reading the pieces, not the whole.</p>'

        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# Build tabs: 5 poems + custom
tab_labels = [k.split(" — ")[0] for k in POEMS.keys()] + ["your poem"]
tabs = st.tabs(tab_labels)

# Preloaded poems
for i, (poem_name, poem_data) in enumerate(POEMS.items()):
    with tabs[i]:
        st.markdown(f"### {poem_name}")
        st.markdown(
            f'<p class="meta" style="font-style:italic;">{poem_data["headline"]}</p>',
            unsafe_allow_html=True,
        )

        # Two columns: poem text + intro commentary
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                f'<div style="font-family: \'Iowan Old Style\', serif; font-size: 17px; '
                f'line-height: 1.7; padding: 16px; background: #1a1815; border-radius: 6px; '
                f'white-space: pre-wrap;">{poem_data["text"]}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="commentary">{poem_data["intro"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### what the model sees")
        # tokenizer override per tab
        tokenizer_name = st.selectbox(
            "tokenizer",
            list(TOKENIZERS.keys()),
            index=list(TOKENIZERS.keys()).index(poem_data["tokenizer"]),
            key=f"tok_{i}",
        )
        show_n = st.checkbox(
            "show semantic neighbors (slow first time)",
            value=False, key=f"neigh_{i}",
        )
        analyze(poem_data["text"], tokenizer_name, show_neighbors=show_n)

        st.markdown("#### findings")
        for title, body in poem_data["findings"]:
            st.markdown(
                f'<div class="finding">'
                f'<div class="finding-title">{title}</div>'
                f'<div style="color:#c8c4b8; font-size:14px; line-height:1.6;">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="commentary" style="margin-top:20px;">'
            f'<strong style="font-style:normal;">takeaway —</strong> {poem_data["takeaway"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

# Custom input tab
with tabs[-1]:
    st.markdown("### your own poem")
    st.markdown(
        '<p class="meta">paste anything — a poem, a sentence, a single word. '
        'the same analysis runs on whatever you give it.</p>',
        unsafe_allow_html=True,
    )

    custom_text = st.text_area(
        "poem",
        value="",
        placeholder="paste a poem here…",
        height=180,
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        custom_tok = st.selectbox("tokenizer", list(TOKENIZERS.keys()), key="custom_tok")
    with c2:
        custom_neigh = st.checkbox(
            "show semantic neighbors (slow first time)",
            value=False, key="custom_neigh",
        )

    if custom_text.strip():
        st.markdown("#### what the model sees")
        analyze(custom_text, custom_tok, show_neighbors=custom_neigh)

        st.markdown("#### what to look for")
        st.markdown(
            '<div class="commentary">'
            'Compare a few tokenizers on the same text. Notice which words stay whole vs. which fragment. '
            'Open the number-line panel — does your poem cluster at the low end (common words) or stretch into the long tail? '
            'If you turn on neighbors, look for words whose nearest semantic relatives are <em>not</em> what you\'d expect — '
            'those are the words where the model is reading something different than you are.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="meta" style="margin-top:24px;">'
            'paste something above to begin.'
            '</p>',
            unsafe_allow_html=True,
        )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p class="meta" style="text-align:center; padding:1rem 0;">'
    'a computational poetry experiment · '
    'tokenizers: tiktoken, transformers · embeddings: all-MiniLM-L6-v2'
    '</p>',
    unsafe_allow_html=True,
)
