# NAVIGATOR — System Architecture (C4)

This document describes NAVIGATOR using the [C4 model](https://c4model.com)
(Context → Containers → Components). The README states the philosophy; this
document encodes it structurally. If the diagrams ever drift from the
philosophy, the diagrams are wrong.

> The code is durable machinery; value is made at reading.

## Architecturally load-bearing invariants

These are the structural commitments. Code reviews should reject changes that
violate them:

1. **The codec is the invertible map and is FROZEN.** All `audio ⇄ code`
   traffic flows through a single adapter. No other component encodes or
   decodes audio. No component fine-tunes the codec.
2. **The Reader is the only substantial trainable component**, plus an
   optional tiny corpus-code model. Training arrows in every diagram point
   into `reader` (and `corpus_model`), nowhere else.
3. **Reading events are dual-purpose.** Every gesture writes to one place
   (`store`) and is then read by *two* downstream consumers: `reader` (for
   training) and `navigate.reader_gradient_step` (for direction in code
   space). That fork is the synthesis — it is what neither parent design
   has alone. Removing either consumer collapses NAVIGATOR.
4. **Returned songs are Reader-carved slices.** No component ships a raw
   decode out to the user as a song. The carving step happens inside `loop`
   between `codec.decode` and the API response.
5. **Provenance is mandatory.** Every `GenerationResult` carries the
   originating `Code`, the full `PerturbOp` chain, and the Reader's
   segmentation. The schemas refuse to construct a result without them.
6. **Genre tags are optional weak features only.** No component branches on
   them. The pipeline must run with tags stripped.

---

## Level 1 — System Context

NAVIGATOR is operated by one person, on one machine. It depends on a small
number of frozen, downloaded model weights and assumes a reverse proxy
handles authentication (no auth in the app itself).

```mermaid
flowchart TB
    user(["Listener-Operator<br/>(builder + user + ultimate reader)"])
    proxy["Reverse Proxy<br/>(handles auth — out of scope)"]
    nav["<b>NAVIGATOR</b><br/>Personal music generation system<br/>(Python, FastAPI, SQLite)"]
    codec["<b>Frozen Codec</b><br/>DAC or EnCodec weights<br/>(downloaded once; never fine-tuned)"]
    gen["<b>Frozen Generator</b> (optional)<br/>MusicGen weights<br/>(downloaded; toggleable)"]
    feats["<b>Frozen Feature Backbone</b> (fallback only)<br/>MERT / CLAP<br/>(used only if codec latents prove poor)"]

    user -- "reads candidates,<br/>gestures (keep/skip/<br/>more-like-this/scrub/thumb)" --> proxy
    proxy -- "localhost HTTP" --> nav
    nav -- "loads, runs (frozen)" --> codec
    nav -. "loads, runs (frozen, optional)" .-> gen
    nav -. "loads, runs (frozen, fallback)" .-> feats

    classDef external fill:#eee,stroke:#888,color:#333
    classDef frozen fill:#f5f5f5,stroke:#888,stroke-dasharray:4 4,color:#333
    classDef system fill:#dde7f5,stroke:#3b6,color:#000
    class user external
    class proxy external
    class codec frozen
    class gen frozen
    class feats frozen
    class nav system
```

Notes:

- The **listener-operator is the ultimate reader.** Their gestures train the
  Reader and point a direction in code space. This is not a separate "admin"
  role: there is one user, who is also the dev.
- Dashed arrows mark **optional** dependencies. The system must run with the
  generator disabled and codec latents as the only feature source.
- The reverse proxy is documented but not built. NAVIGATOR binds `127.0.0.1`
  only.

---

## Level 2 — Containers

NAVIGATOR is a single Python process (FastAPI + in-process PyTorch) plus
file-system persistence. There is no message queue, no separate worker, no
remote DB. One person, one process.

```mermaid
flowchart TB
    user(["Listener-Operator"])

    subgraph nav["NAVIGATOR (one host)"]
      web["<b>Web UI</b><br/>Minimal HTML/JS in browser<br/>captures every gesture as a ReadingEvent"]
      api["<b>API Server</b><br/>FastAPI + PyTorch (in-process)<br/>hosts codec, navigate, reader, loop"]
      sqlite[("<b>SQLite</b><br/>clips, codes, perturb_ops,<br/>reading_events, preference_pairs,<br/>generation_results")]
      fs[("<b>File Store</b><br/>content-addressed audio (wav)<br/>+ code cache (npz/safetensors)")]
    end

    weights[("Frozen model weights<br/>on local disk<br/>(DAC/EnCodec, MusicGen, MERT)")]

    user -- "HTTPS via reverse proxy" --> web
    web -- "JSON over HTTP<br/>/encode /generate /events /ab/pair /status" --> api
    api -- "SQL (metadata + events)" --> sqlite
    api -- "read/write (audio + codes)" --> fs
    api -- "load + forward pass<br/>(frozen)" --> weights

    classDef external fill:#eee,stroke:#888,color:#333
    classDef frozen fill:#f5f5f5,stroke:#888,stroke-dasharray:4 4,color:#333
    classDef container fill:#dde7f5,stroke:#3b6,color:#000
    classDef datastore fill:#fff4d6,stroke:#a80,color:#000
    class user external
    class weights frozen
    class web,api container
    class sqlite,fs datastore
```

Container responsibilities:

| Container | Responsibility | Trainable? |
|---|---|---|
| **Web UI** | Cold-start A/B reading; warm-mode player with "more/less like this part" slider, scrub, thumbs. Every interaction emits a `ReadingEvent` to the API. No localStorage / sessionStorage — server-side state only. | No |
| **API Server** | HTTP surface + orchestration of the cold/warm loop. Owns the in-process Reader. Loads frozen models on startup. | The Reader inside it is. |
| **SQLite** | Metadata and the *durable record of reading*. Reading events and preference pairs live here. Generation provenance lives here. | No |
| **File Store** | Content-addressed audio (sha256-sharded) and code tensors. Codes are large; SQLite holds references only. | No |
| **Frozen weights** (external) | Downloaded once; never written. License table belongs in README. | No (frozen) |

---

## Level 3 — Components inside the API Server

The Python package is `src/navigator/`. Each file is one component. The
diagram shows their dependencies and — more importantly — the **fork of
reading events into two downstream consumers**, which is the synthesis.

```mermaid
flowchart LR
    subgraph http["HTTP surface"]
      api["<b>api</b><br/>FastAPI endpoints"]
    end

    subgraph orch["Orchestration"]
      loop["<b>loop</b><br/>cold/warm orchestrator"]
      bootstrap["<b>bootstrap</b><br/>cold-start active learning"]
    end

    subgraph d_side["D — moves (frozen)"]
      codec["<b>codec</b><br/>frozen DAC/EnCodec<br/>encode / decode / latents"]
      navigate["<b>navigate</b><br/>noise / interpolate / splice /<br/>mask_regen / corpus_sample /<br/><b>reader_gradient_step</b>"]
      generator["<b>generator</b> (optional)<br/>frozen MusicGen<br/>mask_regen / continuation"]
      corpus_model["<b>corpus_model</b> (optional)<br/>tiny GMM/VAE/AR<br/>over codes"]
    end

    subgraph c_side["C — judgement (trainable)"]
      reader["<b>reader</b><br/>value head + boundary head<br/>over frozen features<br/><i>(only substantial trainable)</i>"]
    end

    subgraph data["Data model + persistence"]
      schemas["<b>schemas</b><br/>Clip / Code / PerturbOp /<br/>ReadingEvent / PreferencePair /<br/>GenerationRequest / GenerationResult"]
      store["<b>store</b><br/>SQLite + content-addressed FS"]
      settings["<b>settings</b><br/>config.yaml → pydantic"]
    end

    api --> loop
    api -- "writes every gesture" --> store
    api --> bootstrap

    loop -- "asks for moves" --> navigate
    loop -- "decode candidates" --> codec
    loop -- "asks for judgement<br/>(segment + score)" --> reader
    loop -- "carves slice, writes<br/>GenerationResult with provenance" --> store

    bootstrap --> navigate
    bootstrap --> codec
    bootstrap -- "writes ReadingEvent +<br/>PreferencePair (both codes)" --> store

    navigate --> codec
    navigate -. "if generator on" .-> generator
    navigate -. "if corpus model on" .-> corpus_model

    %% THE SYNTHESIS — reading events fork into two consumers
    store == "preference pairs<br/>(both codes)" ==> reader
    store == "preferred / dispreferred<br/>code positions" ==> navigate

    reader -. "consumes frozen features" .-> codec

    classDef frozen fill:#f5f5f5,stroke:#888,stroke-dasharray:4 4,color:#333
    classDef trainable fill:#e8f5e0,stroke:#3a3,color:#000
    classDef data fill:#fff4d6,stroke:#a80,color:#000
    classDef orch fill:#dde7f5,stroke:#3b6,color:#000
    class codec,generator frozen
    class reader,corpus_model trainable
    class store,schemas,settings data
    class api,loop,bootstrap orch
    class navigate orch
```

Component responsibilities (1-line each — full docstrings live in the source):

| Component | Responsibility | Frozen / trainable |
|---|---|---|
| `settings` | Load `config.yaml` into a pydantic tree. | n/a |
| `schemas` | All data types. The design lives here. | n/a |
| `store` | SQLite + content-addressed audio + code cache. | n/a |
| `codec` | The invertible map. `encode` / `decode` / `latents`. Round-trip ≈ identity. | FROZEN |
| `navigate` | Code-space ops. Houses `reader_gradient_step` (the synthesis op). | FROZEN-arithmetic |
| `reader` | Value head + boundary head over frozen features. Constitutes texts (segments) and confers value. | **TRAINABLE** |
| `generator` | Optional MusicGen for `mask_regen` / continuation. Toggleable. | FROZEN |
| `corpus_model` | Optional tiny model over codes. `sample()` / `score()`. | **TRAINABLE** (few MB) |
| `bootstrap` | Cold-start active learning: diverse codes → forced A/B → preference pairs. | n/a |
| `loop` | The core orchestrator. Cold mode = A/B; warm mode = navigate→decode→read→rank→carve. | n/a |
| `api` | FastAPI HTTP surface. Captures gestures. No auth (reverse proxy handles it). | n/a |

The thick `==>` arrows from `store` to `reader` AND `navigate` are the
**synthesis arrow** — the architectural feature that makes NAVIGATOR more
than the sum of its parents. Every reading event writes one row in `store`
and is consumed twice: once by `reader.train` (to improve judgement) and
once by `navigate.reader_gradient_step` (to set the next direction in code
space). Because both consumers read the same source of truth, the Reader's
judgement and the navigator's direction can never disagree about what
"preferred" means.

---

## Key flows

### Cold start (M3 → M4)

```mermaid
sequenceDiagram
    autonumber
    actor U as Listener-Operator
    participant W as Web UI
    participant A as api
    participant B as bootstrap
    participant N as navigate
    participant C as codec
    participant S as store
    participant R as reader

    U->>W: "start"
    W->>A: GET /ab/pair
    A->>B: next pair
    B->>N: diverse code sampling (no anchor)
    N-->>B: code_X, code_Y
    B->>C: decode(code_X), decode(code_Y)
    C-->>B: audio_X, audio_Y
    B->>S: persist Clips + Codes (no value yet — Code carries no value)
    B-->>A: {clip_X, clip_Y, codes}
    A-->>W: pair
    W->>U: play X / play Y / choose
    U->>W: "X"
    W->>A: POST /events {gesture: ab_choice, clip: X, code: code_X, counterpart: Y/code_Y}
    A->>S: write 2 ReadingEvents + 1 PreferencePair (both codes recorded)
    Note over S,R: After target_pairs reached, train_reader.py fits the value head.<br/>If held-out pairwise accuracy ≥ gate, mode → warm.
```

### Warm loop with the synthesis arrow (M5 → M6)

```mermaid
sequenceDiagram
    autonumber
    actor U as Listener-Operator
    participant W as Web UI
    participant A as api
    participant L as loop
    participant C as codec
    participant N as navigate
    participant R as reader
    participant S as store

    U->>W: "more like clip_anchor"
    W->>A: POST /generate {mode: like, anchor: clip_anchor, strength: 0.3}
    A->>L: GenerationRequest
    L->>C: encode(clip_anchor) → code_anchor
    L->>S: load preference pairs (preferred/dispreferred codes)
    L->>N: navigate(code_anchor, strength) + reader_gradient_step(pref_codes, disp_codes)
    N-->>L: K candidate codes (count × candidate_multiplier)
    L->>C: decode each candidate
    C-->>L: K decoded audios
    L->>R: segment + score each
    R-->>L: ScoredSegments per candidate
    L->>L: rank, carve top `count` slices from highest-scoring segments
    L->>S: persist Clips (slice + raw) + Codes + PerturbOp chain + GenerationResult
    L-->>A: results
    A-->>W: GenerationResult[] — SLICES, not raw decodes
    W->>U: play slices
    U->>W: gesture (e.g. "more_like_this" with time range)
    W->>A: POST /events {clip, code, segment, gesture}
    A->>S: write ReadingEvent (anchored to the candidate's CODE)
    Note over S,R: Reading event is now read by reader.train (judgement)<br/>AND by navigate.reader_gradient_step (next direction in code space) — the synthesis.
```

The "synthesis payoff" sentence in step 16 is the architectural punchline:
the same row in `store` simultaneously improves the Reader and tells the
navigator where to move next, with no separate training loop and no
hand-built reward model.

---

## Cross-cutting concerns

### Provenance

Every generated artefact carries its full lineage:

- `Clip.lineage.parent_clip_ids` — human-meaningful anchors (`--like A`, `A↔B`)
- `Clip.lineage.parent_code_ids` — the codes that seeded perturbation
- `Clip.lineage.perturb_op_ids` — every move applied, in order
- `GenerationResult.code` — the candidate code (the code IS the durable mark)
- `GenerationResult.perturb_op_chain` — replayable record of how it was reached
- `GenerationResult.reader_segmentation` — the Reader's carving with per-segment scores

This is checked at schema construction time; you cannot persist a result
without it.

### Frozen vs trainable (the trust boundary)

| Frozen | Trainable |
|---|---|
| `codec` (DAC/EnCodec) | `reader` (value + boundary heads) |
| `generator` (MusicGen, optional) | `corpus_model` (optional, few MB) |
| `features` fallback (MERT/CLAP, if used) | — |

Any PR that adds a `.train()` call or `requires_grad=True` on a frozen
component must stop and re-read the README. If a future need pushes you
to fine-tune the codec, that's a different system.

### Storage layout

```
data/
  audio/<sha256[:2]>/<sha256>.wav        # content-addressed audio
  codes/<code_id>.npz                    # token/latent tensors
  navigator.sqlite                       # everything else
```

The two file stores are append-only by convention; nothing in NAVIGATOR
overwrites a content-addressed path. SQLite rows are mutable for status
fields (e.g. clip metadata) but never for `ReadingEvent` (events are facts
about the past) or `PerturbOp` (provenance).

### Security & auth

- NAVIGATOR binds `127.0.0.1` only.
- The app has no authentication. A reverse proxy handles it.
- There is one user (the operator). No multi-tenant logic exists or is planned.

### Compute

| Step | M4-Pro (CPU/MPS) | 24 GB GPU (CUDA) |
|---|---|---|
| Codec encode (a few seconds of audio) | seconds | sub-second |
| Codec decode | seconds | sub-second |
| Reader train (after cold start, ~40 pairs) | seconds | sub-second |
| Reader train (warm, ~3 000 pairs) | a few minutes | seconds |
| Optional generator forward (MusicGen-small) | tens of seconds | seconds |

(These are rough order-of-magnitude expectations. The README's M8 walkthrough
will quote measured numbers.)

---

## Non-goals (architecturally — not "later")

These are out of scope of NAVIGATOR as designed. If a future need pushes
toward any of them, that is a different system; do not extend NAVIGATOR.

- **Training a codec from scratch.** Cluster + team territory.
- **Training a generator from scratch.** Same.
- **A "proper D" pipeline with evolved/MDL notation.** That is a different
  research project; NAVIGATOR is the runnable fusion.
- **Multi-user / multi-tenant.** One operator. No accounts table.
- **Cloud-managed services.** One host. SQLite + filesystem.
- **A taste corpus seeded by the developer.** Start blank. Taste comes into
  being through use.
- **A "quality gate" that filters whole decodes by yes/no.** That collapses
  the Reader from constitutor to filter; the design is broken.
