# poem → color

A focused demo for a computational poetry meetup. Four poems where tokenization changes what a language model can see, plus a custom-input mode for trying your own text.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run will download model weights (~500MB total: tiktoken vocab files, BERT, LLaMA tokenizer, MiniLM sentence transformer, PyTorch). Subsequent runs are fast.

## Deploy to Streamlit Community Cloud

This is the intended deployment path.

1. Push this folder to a public GitHub repo (e.g. `alicia/poem-color`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click "New app" → pick the repo → main branch → `app.py`
4. Click Deploy. First build takes ~5 minutes (installing dependencies + downloading models). Subsequent loads are fast because Streamlit caches the resources.

Your app will be live at `https://<your-username>-poem-color-<hash>.streamlit.app` (you can rename the URL slug in app settings).

### Notes for Streamlit Cloud

- Free tier gives 1GB RAM, which is enough but tight. The MiniLM model is ~80MB, BERT tokenizer ~200MB, plus PyTorch overhead.
- The `@st.cache_resource` decorators are critical — without them, every rerun reloads the models and the app will crash.
- The neighbor corpus loads from a GitHub raw URL on first use (~6K common English words). If GitHub rate-limits, the app falls back to a small hardcoded list.
- LLaMA tokenizer is from `hf-internal-testing/llama-tokenizer` (un-gated mirror); falls back to T5 (also SentencePiece-based) if that fails.

## What's in here

Four preloaded poems, each with:
- the text
- a written intro framing the question
- live tokenization (with selectable tokenizer)
- the colored token strip (embedding → RGB)
- a number-line view showing where IDs land in the vocab
- optional: semantic neighbors of each token
- 3 written findings specific to that poem
- a takeaway paragraph

Plus a **your poem** tab with the same analysis on arbitrary input.

The four poems and their findings:

- **Williams — The Red Wheelbarrow.** "Barrow" reads as a burial mound, not half of a wheelbarrow. The model cannot see the picture.
- **cummings — anyone lived in a pretty how town.** The seasons form a tight chromatic cluster; cummings's grammatical inversions are invisible to the embedding.
- **Bashō — old pond.** Short enough to compare all four tokenizers at a glance. The number-line view shows each tokenizer's underlying philosophy.
- **Dickinson — Because I could not stop for Death.** Even cl100k splits "Immortality" into pieces. Tokenizer vocabularies reflect their training corpus, not literary diction.

## Tech

- **Tokenizers**: `tiktoken` (cl100k_base for GPT-4), `transformers` (LLaMA SentencePiece, BERT WordPiece), Python builtins for character-level
- **Embeddings**: `sentence-transformers` with all-MiniLM-L6-v2
- **Color**: PCA projection of MiniLM embeddings onto 3 anchor-defined axes (warm/organic/cool ish)
- **Plots**: matplotlib
