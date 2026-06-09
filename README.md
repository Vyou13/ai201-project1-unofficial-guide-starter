# The Unofficial Guide — Project 1

A Retrieval-Augmented Generation (RAG) system that answers plain-language questions about NYU
Computer Science professors using real student reviews from Rate My Professors. Ask
*"Is Alan Siegel a good professor for algorithms?"* and get a grounded, cited answer drawn only
from the reviews collected — never from the model's general knowledge.

**Pipeline:** `.txt` reviews → clean → chunk → embed (all-MiniLM-L6-v2) → ChromaDB → top-5 retrieval → Groq LLM (grounded) → Gradio UI.

```
python ingest.py documents/docs   # inspect chunks
python embed.py                    # build the vector store (ChromaDB)
python test_pipeline.py            # run all 5 eval questions + refusal test
python app.py                      # launch the Gradio interface (http://localhost:7860)
```

---

## Domain

Student reviews of **Computer Science professors at New York University (NYU)**, sourced from
Rate My Professors (ratemyprofessors.com).

This knowledge is valuable because it captures the real student experience — exam difficulty,
grading style, how helpful a professor is outside class, workload, and teaching style — none of
which appear in NYU's official course catalog or department website. A student choosing between
two professors for the same CS course has no official way to compare them; the information is
scattered across dozens of individual reviews. This system makes that knowledge searchable and
returns a single grounded, cited answer instead of forcing the student to read 60 reviews by hand.

---

## Document Sources

10 NYU CS professors, one `.txt` file per professor, each containing the professor's individual
student reviews (rating, difficulty, course, date, and review text) copied from their Rate My
Professors page. (Rate My Professors is JavaScript-rendered and blocks scrapers, so reviews were
collected manually into plain-text files — a normal and expected workflow per the project notes.)

| #  | Source | Type | URL or file path |
|----|--------|------|------------------|
| 1  | Craig Kapp — Intro CS (CSCI-UA 2), 159 reviews, 4.8/5 | RMP reviews (.txt) | https://www.ratemyprofessors.com/professor/1579749 → `documents/docs/craig_kapp.txt.txt` |
| 2  | Joshua Clayton — CS (CSCI-UA 4), 67 reviews, 4.7/5 | RMP reviews (.txt) | https://www.ratemyprofessors.com/professor/1808953 → `documents/docs/joshua_clayton.txt.txt` |
| 3  | Ben Goldberg — Data Structures (CSCI-UA 201), 65 reviews, 4.7/5 | RMP reviews (.txt) | https://www.ratemyprofessors.com/professor/216065 → `documents/docs/ben_goldberg.txt.txt` |
| 4  | Anirudh Sivaraman — Networking (CS 480), 18 reviews, 4.6/5 | RMP reviews (.txt) | https://www.ratemyprofessors.com/professor/2329940 → `documents/docs/anirudh_sivaraman.txt.txt` |
| 5  | Evan Korth — CS102, 114 reviews, 3.5/5 (polarizing) | RMP reviews (.txt) | https://www.ratemyprofessors.com/professor/272737 → `documents/docs/evan_korth.txt.txt` |
| 6  | Alan Siegel — Algorithms (CSCI 310), 69 reviews, 2.7/5 | RMP reviews (.txt) | https://www.ratemyprofessors.com/professor/1941844 → `documents/docs/alan_siegel.txt.txt` |
| 7  | Michael Tao — Intro CS (CSCI-UA 101), 18 reviews, 2.8/5 | RMP reviews (.txt) | https://www.ratemyprofessors.com/professor/2910630 → `documents/docs/michael_tao.txt.txt` |
| 8  | Candido Cabo — Intro CS (CSCI-UA 101), 17 reviews, mixed | RMP reviews (.txt) | https://www.ratemyprofessors.com/professor/2567373 → `documents/docs/candido_cabo.txt.txt` |
| 9  | Patrick Cousot — PL / Static Analysis (CSCI 3140), 20 reviews, 5.0/5 | RMP reviews (.txt) | https://www.ratemyprofessors.com/professor/1762046 → `documents/docs/patrick_cousot.txt.txt` |
| 10 | Alan Amin — CS102, 7 reviews, 3.9/5 | RMP reviews (.txt) | https://www.ratemyprofessors.com/professor/2956463 → `documents/docs/alan_amin.txt.txt` |

The sources were deliberately chosen for **rating diversity** — highly-rated professors (Kapp 4.8,
Cousot 5.0), polarizing ones (Korth 3.5), and low-rated ones (Siegel 2.7, Tao 2.8) — so the system
must handle genuinely positive, negative, and mixed evidence rather than only praise.

---

## Chunking Strategy

Implemented in [`chunk_text()` in ingest.py](ingest.py#L37).

**Chunk size:** 400 characters (~60–80 words), with a 60-character tolerance (hard limit 460).

**Overlap:** 50 characters (only applied when a single review exceeds the hard limit).

**Preprocessing before chunking:** Each file starts with a `#`-prefixed metadata header
(`# Professor:`, `# RMP URL:`, etc.). [`clean_text()`](ingest.py#L26) strips that header block so
only the review content is chunked, while [`parse_header()`](ingest.py#L15) extracts the professor
name and RMP URL into metadata. Because the documents are plain text (not scraped HTML), there are
no tags, nav bars, or ads to remove.

**Why these choices fit the documents:** Each RMP page is a stack of short, self-contained reviews
(typically 1–3 sentences, ~250 chars) separated by blank lines. So the chunker **splits on `\n\n`
first** and keeps each review as one whole chunk whenever it fits under the 460-char hard limit —
this means *one chunk ≈ one complete student opinion*, which is exactly the unit a query like
"what do students say about X" wants to match. The 60-char tolerance prevents a 410-char review
from being split into a 399-char chunk plus a useless 11-char orphan. Only genuinely long reviews
(600+ chars) get character-split at word boundaries, with 50-char overlap so a thought spanning the
split is still retrievable from either half.

**Final chunk count:** **56 chunks** across 10 files. Length stats: min 111, max 460, avg 258
characters. No fragments under 80 chars except deliberate overlap tails. (Comfortably inside the
project's healthy 50–2,000 range — small enough that each embedding carries one clear opinion,
large enough to stay meaningful on its own.)

### Sample chunks (5 labeled, with source)

**Sample 1 — `craig_kapp.txt.txt` (120 chars):**
> Rating: 5/5 | Difficulty: 2/5 | Course: COMPPROG | Date: Mar 25th, 2026
> Best prof ever. Amazing lectures. Inspirational.

**Sample 2 — `alan_siegel.txt.txt` (218 chars):**
> Rating: 2/5 | Difficulty: 5/5 | Course: CSCI 310 | Date: Dec 15th, 2025
> Algorithms with Siegel is brutal. His exams are impossibly hard and he doesn't give partial credit. The class average was failing until the curve.

**Sample 3 — `ben_goldberg.txt.txt` (250 chars):**
> Rating: 5/5 | Difficulty: 3/5 | Course: CSCI-UA 201 | Date: Dec 20th, 2025
> Ben is amazing for Data Structures. He explains complex topics like trees and graphs so clearly. Lots of coding but that's how you learn. Always available during office hours.

**Sample 4 — `candido_cabo.txt.txt` (389 chars):**
> Rating: 5/5 | Difficulty: 3/5 | Course: CSCI-UA 101 | Date: Jan 24th, 2026
> Prof Cabo really deserves better reviews. He has nice, informative slides for every lecture, and gives lots of examples for each class. His homeworks help you learn a lot. His exams are a bit tricky though, but if you prepare well and bring a nice cheatsheet, you'll do fine. He also gives big curves on each exam!

**Sample 5 — `michael_tao.txt.txt` (329 chars):**
> Rating: 2/5 | Difficulty: 4/5 | Course: CSCI-UA 101 | Date: Feb 2nd, 2026
> Tests are extremely long, quite difficult compared to covered lectures and content. The test questions are hard to even understand. Most of the class did not finish Exam II or III. Very hard tests, assignments easy. Tough grader. Accessible outside class.

Each chunk is a complete, standalone student opinion that includes its rating, difficulty, course,
and date — readable and answerable without needing the surrounding text.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dimensional embeddings).
Stored in a local **ChromaDB** collection configured for cosine distance
([`embed.py`](embed.py#L44)). Retrieval uses **top-k = 5** ([`retrieve()` in query.py](query.py#L38)).

It runs entirely locally with no API key, no rate limits, and no per-token cost, which fits a
free, self-contained student project. At 384 dimensions it embeds all 56 chunks in ~1 second on
CPU, and its quality is more than enough for short opinion text.

**Production tradeoff reflection:** If I were deploying this for real users and cost weren't a
constraint, I'd weigh:

- **Context length:** MiniLM truncates at 256 tokens. That's fine for my ~250-char reviews, but
  if I ingested long-form guides or full forum threads I'd want a model with a larger window like
  OpenAI's `text-embedding-3-small` (8,192 tokens) so I wouldn't have to chunk as aggressively and
  could keep whole documents intact.
- **Multilingual support:** NYU has many international students who sometimes review in other
  languages. MiniLM is English-only; `multilingual-e5-large` would embed non-English reviews into
  the same space as the English ones so a query could match both.
- **Domain accuracy:** A stronger general model like `bge-large-en-v1.5` scores higher on MTEB
  retrieval benchmarks and still runs locally — better at catching that "brutal exams" and
  "impossibly hard tests" mean the same thing — at the cost of ~1.5 GB RAM and higher latency.
- **Local vs. API latency/cost:** Local MiniLM has zero network latency and zero cost but uses my
  CPU/RAM; an API model offloads compute and gives better accuracy but adds per-token cost, network
  latency, and a hard dependency on a third-party service being up. For a production app I'd likely
  pick `text-embedding-3-small` (API) for accuracy + context length, accepting the cost.

---

## Retrieval Test Results

Run via [`test_pipeline.py`](test_pipeline.py) against the live ChromaDB collection (cosine
distance — **lower = more similar**, k=5).

### Query 1 — "What do students say about Craig Kapp's exams and grading?"
| # | distance | professor | source | chunk preview |
|---|----------|-----------|--------|---------------|
| 1 | 0.387 | Alan Amin | alan_amin.txt.txt | "The quizzes and exams are really quite hard so be prepared…" |
| 2 | 0.431 | Michael Tao | michael_tao.txt.txt | "The exams ARE hard and stressful, as other comments said…" |
| 3 | 0.458 | Michael Tao | michael_tao.txt.txt | "His exams are long and tough. He requires hand-written code…" |
| 4 | 0.465 | Alan Amin | alan_amin.txt.txt | "Nice guy. Gives quizzes and homework assignments weekly…" |
| 5 | 0.472 | Candido Cabo | candido_cabo.txt.txt | "I admit lectures are boring, but he is thorough…" |

⚠️ **This is a retrieval failure** — Craig Kapp does not appear in his own query's top 5. Analyzed
in the [Failure Case](#failure-case-analysis) section below.

### Query 2 — "Is Alan Siegel a good professor for algorithms?"
| # | distance | professor | source | chunk preview |
|---|----------|-----------|--------|---------------|
| 1 | 0.355 | Alan Siegel | alan_siegel.txt.txt | "Unpopular opinion but I liked Siegel. Yes it's hard. Yes the exams are rough. But you learn…" |
| 2 | 0.422 | Alan Siegel | alan_siegel.txt.txt | "Algorithms with Siegel is brutal. His exams are impossibly hard and he doesn't give partial credit…" |
| 3 | 0.460 | Alan Siegel | alan_siegel.txt.txt | "He knows algorithms inside and out. But he's not great at teaching them…" |
| 4 | 0.487 | Alan Siegel | alan_siegel.txt.txt | "Siegel is a mixed bag. He's clearly brilliant but can be condescending…" |
| 5 | 0.593 | Craig Kapp | craig_kapp.txt.txt | "Only 1 month in and I already know he's the best professor I'll ever have at NYU…" |

**Why these chunks are relevant:** The top 4 results are all Alan Siegel reviews about CSCI 310
(Algorithms), with low distances (0.36–0.49). They directly cover the question — they discuss his
difficulty ("brutal", "impossibly hard"), his teaching ("not great at teaching them"), and offer
both a defender ("I liked Siegel… you learn") and critics ("condescending"), which is exactly the
balanced evidence needed to judge whether he's "a good professor." The query shares the words
"Siegel" and "algorithms" with the reviews, *and* the embedding correctly grouped the difficulty
language. The 5th result (Kapp, distance 0.593) is off-topic noise — the gap from 0.49 to 0.59
marks where relevance drops off.

### Query 3 — "Which professor would be best for a beginner with no programming experience?"
| # | distance | professor | source | chunk preview |
|---|----------|-----------|--------|---------------|
| 1 | 0.527 | Craig Kapp | craig_kapp.txt.txt | "Professor Kapp was incredible! He gave me such a good intro to the computer programming world…" |
| 2 | 0.544 | Craig Kapp | craig_kapp.txt.txt | "Best CS professor at NYU hands down. Actually cares if you learn. Lectures are engaging…" |
| 3 | 0.553 | Michael Tao | michael_tao.txt.txt | "Not a bad professor. Lectures can be boring but it's good to study them for exams…" |
| 4 | 0.557 | Michael Tao | michael_tao.txt.txt | "The exams ARE hard and stressful…" |
| 5 | 0.566 | Craig Kapp | craig_kapp.txt.txt | "Really good professor. Only downside is the flipped classroom style…" |

**Why these chunks are relevant:** The query never mentions any professor by name, yet semantic
search surfaced **Craig Kapp** in 3 of the top 5 slots — including the review that literally says
*"gave me such a good intro to the computer programming world."* This demonstrates why semantic
search beats keyword search here: the query word "beginner" never appears in the reviews, but the
embedding maps "beginner with no programming experience" close to "good intro to programming" and
"actually cares if you learn." The Michael Tao chunks (distance ~0.55) are weaker matches that
slipped in because they mention intro CS (CSCI-UA 101) but talk about hard exams, not beginners —
a sign the distances here (all 0.52–0.57) are borderline, unlike the tight Siegel cluster above.

*(Additional verified runs for Q2 "Michael Tao complaint" and Q4 "Ben Goldberg availability" are
included in the Evaluation Report below.)*

---

## Grounded Generation

Implemented in [`ask()` in query.py](query.py#L84) with the system prompt at
[query.py:74](query.py#L74).

**System prompt grounding instruction (verbatim):**
> You are a helpful assistant that answers questions about NYU Computer Science professors based
> ONLY on student reviews provided to you as context.
>
> Rules you must follow:
> 1. Answer ONLY from the context below. Do not use outside knowledge.
> 2. At the end of your answer, cite the source files you used, like: Sources: craig_kapp.txt, ben_goldberg.txt
> 3. If the context does not contain enough information to answer the question confidently, respond with exactly: "I don't have enough information to answer that based on the available reviews."
> 4. Do not speculate or fill gaps with assumptions.
> 5. Be concise and direct.

**How grounding is enforced — prompt design *and* pipeline structure:**

1. **Retrieval-first context injection.** `ask()` calls `retrieve()` first and builds the user
   message *only* from the top-5 chunks, each prefixed with its source and professor
   (`[1] Source: … | Professor: …`). The model never sees anything but the retrieved reviews.
2. **Explicit "ONLY from context" instruction** plus a **mandatory refusal string** — rule 3 forces
   the exact sentence *"I don't have enough information…"* when the context is insufficient, so the
   model declines instead of inventing an answer from training knowledge.
3. **Low temperature (0.2)** at [query.py:121](query.py#L121) minimizes creative drift and keeps
   answers tight to the provided text.

**How source attribution is surfaced:** Two ways, belt-and-suspenders. (a) The model is instructed
to cite source files inline (rule 2), and (b) **the pipeline appends sources programmatically** —
`ask()` returns a deduplicated `sources` list built directly from the retrieved chunks'
metadata ([query.py:127-134](query.py#L127-L134)), so attribution is *guaranteed* even if the LLM
forgets to cite. The Gradio UI displays these in a dedicated "Sources" panel with the professor
name and RMP URL ([app.py:26-36](app.py#L26-L36)).

---

## Example Responses

Live outputs from the running system (Groq `llama-3.3-70b-versatile`, temperature 0.2).

**Example 1 — "Is Alan Siegel a good professor for algorithms?"**
> Reviews are mixed about Professor Alan Siegel for algorithms (CSCI 310). Some students
> appreciate his deep knowledge and fair curving, despite the high difficulty level. However,
> others find him not great at teaching and requiring self-study from the textbook. He can also
> come across as condescending.
>
> **Sources:** alan_siegel.txt.txt

Every claim here traces to a retrieved Siegel review — the "brilliant but condescending", "not
great at teaching", and "self-study from the textbook" points all appear verbatim in the corpus.

**Example 2 — "Which professor would be best for a beginner with no programming experience?"**
> Based on the reviews, Professor Craig Kapp would be the best for a beginner with no programming
> experience. One review mentions that he gave a "good intro to the computer programming world"
> and another states that he "explains complex topics simply", indicating that he is well-suited
> for beginners.
>
> **Sources:** craig_kapp.txt.txt

The model quotes the actual review text ("good intro to the computer programming world") rather
than asserting general knowledge about intro CS.

**Out-of-scope query — "What is the best pizza place near NYU?"**
> I don't have enough information to answer that based on the available reviews.

✅ The system refuses exactly as instructed (rule 3 of the system prompt) instead of inventing a
restaurant recommendation from the model's training data — grounding working as designed.

---

## Query Interface

A **Gradio web UI** ([`app.py`](app.py)), launched with `python app.py` at `http://localhost:7860`.

**Input fields:**
- **"Your Question"** — a free-text box where the user types a plain-language question (e.g.
  *"What do students say about Craig Kapp's exams?"*). Submits on button click or Enter. Six
  one-click example questions are provided below the box.
- **"Search mode"** — a radio toggle between **Smart (filter + hybrid)** (default — the stretch
  retrieval that auto-detects a professor and blends BM25 + semantic) and **Semantic only** (the
  baseline, useful for demonstrating the Craig Kapp retrieval failure).

**Output fields:**
- **"Answer"** — the grounded, LLM-generated answer (or the refusal sentence if the reviews don't
  cover the question).
- **"Sources"** — a bulleted list of the documents the answer drew from, each showing the professor
  name, source filename, and a clickable RMP URL.

**Sample interaction transcript:**
```
[Your Question]  Is Alan Siegel a good professor for algorithms?
[Search mode]    Smart (filter + hybrid)

[Answer]   Reviews are mixed about Professor Alan Siegel for algorithms (CSCI 310). Some
           students appreciate his deep knowledge and fair curving, despite the high
           difficulty level. However, others find him not great at teaching and requiring
           self-study from the textbook. He can also come across as condescending.

[Sources]  • Alan Siegel — alan_siegel.txt.txt
             https://www.ratemyprofessors.com/professor/1941844
```

---

## Evaluation Report

The 5 test questions from [`planning.md`](planning.md), run through the **baseline semantic
pipeline** (`python test_pipeline.py`). Results below are verbatim live output. (The stretch
"smart" mode fixes Q1 — see the [Stretch Features](#stretch-features) section.)

| # | Question | Expected answer | System response (baseline semantic) | Retrieval quality | Response accuracy |
|---|----------|-----------------|-----------------|-------------------|-------------------|
| 1 | What do students say about Craig Kapp's exams and grading? | Tests/quizzes are fair with no tricks, extra credit, easy to do well if you do the work; difficulty ~2–3/5. | *"I don't have enough information to answer that based on the available reviews."* | **Off-target** — top 5 are Alan Amin & Michael Tao, **no Kapp** | **Inaccurate** — corpus *does* contain the answer; retrieval never surfaced it, so the system (correctly, given its context) refused (see failure case) |
| 2 | What is the main complaint about Michael Tao's intro CS class? | Exams are extremely long, confusing, harder than lectures; students don't finish; half fail the 2nd midterm. | "The main complaint … is that his tests are extremely challenging, making it difficult for students who are not strong in CS to prepare and perform well." | **Partially relevant** — #1 correctly Michael Tao (dist 0.363), #2–5 drift to Sivaraman/Korth/Goldberg | **Partially accurate** — captures "extremely challenging tests" but misses the specifics (not finishing exams, half failing the 2nd midterm) because those chunks weren't in the top 5 |
| 3 | Is Alan Siegel a good professor for algorithms? | Tough grader, very hard class; some find him brilliant/caring. | "Reviews are mixed … some appreciate his deep knowledge and fair curving … others find him not great at teaching … can come across as condescending." | **Relevant** — top 4 all Siegel/CSCI 310, dist 0.36–0.49 | **Accurate** — faithfully reflects the mixed reviews; cites only `alan_siegel.txt` |
| 4 | What do students say about Ben Goldberg's availability and responsiveness? | Responds quickly, nice about extensions, accessible outside class; 4.7/5. | "… very responsive on Ed and in office hours, and … gives extensions when needed." | **Partially relevant** — top 2 are Goldberg (dist 0.49, 0.54); #3–5 drift to other profs | **Accurate** — directly answers availability/responsiveness from Goldberg reviews |
| 5 | Which professor would be best for a beginner with no programming experience? | Craig Kapp or Candido Cabo — Kapp "great intro"; Cabo great at explaining to beginners. | "… Professor Craig Kapp … gave a 'good intro to the computer programming world' and … 'explains complex topics simply' …" | **Relevant** — Kapp in 3 of top 5 incl. "such a good intro to programming" | **Partially accurate** — correctly recommends Kapp with quoted evidence, but misses Candido Cabo (his beginner-friendly reviews never ranked in the top 5) |

**Summary:** 2 accurate (Q3, Q4), 2 partially accurate (Q2, Q5), 1 inaccurate (Q1). The two
"partial"/"inaccurate" results share a root cause — relevant chunks falling outside the top 5
because many professors share vocabulary ("hard exams", "office hours") — which is exactly what
the stretch hybrid + metadata-filter retrieval addresses.

---

## Failure Case Analysis

**Question that failed:** Q1 — *"What do students say about Craig Kapp's exams and grading?"*

**What the system returned:** The top 5 retrieved chunks were **Alan Amin (×2), Michael Tao (×2),
and Candido Cabo (×1)** — **Craig Kapp did not appear at all**, despite being the subject of the
question. With no Kapp reviews in its context, the LLM correctly **refused**: *"I don't have enough
information to answer that based on the available reviews."* This is the right behavior given the
bad context (grounding prevented a hallucination), but it's still a *system failure* because the
corpus genuinely contains the answer — Kapp's reviews include "Tests and quizzes are fair, no
tricks… Extra credit. Clear grading criteria." The closest retrieved match was an Alan Amin review
at distance 0.387.

**Root cause (tied to the embedding/retrieval stage):** `all-MiniLM-L6-v2` embeds on *semantic
content*, and the dominant content words in this query are **"exams and grading" + difficulty**, not
the name "Craig Kapp." Kapp's actual reviews are short and praise-focused
(*"Best prof ever. Amazing lectures. Inspirational."*) and rarely dwell on exam mechanics — so they
embed far from a query about exams. Meanwhile, Alan Amin's and Michael Tao's reviews are *all about*
hard quizzes and tough exams, so they cluster tightly around the query vector. The professor's name
is just 2 of ~10 tokens and gets washed out by the heavier topical signal. This is exactly the risk
flagged as **Anticipated Challenge #2 in planning.md** ("the professor's name is a weak retrieval
signal"). It is a *retrieval* failure, not a generation failure — the LLM never even receives a Kapp
review to ground on, so no prompt tuning could fix it.

**What I would change to fix it — and did:** I implemented **both** fixes as stretch features (see
below). Metadata filtering detects "Kapp" in the query and restricts retrieval to his reviews;
hybrid search additionally weights the exact surname token via BM25. In **smart mode** the same
query now retrieves 5/5 Craig Kapp chunks and the system answers correctly: *"Craig Kapp's tests
and quizzes are fair, with no tricks… clear grading criteria… there's even extra credit available."*
(Sources: craig_kapp.txt.txt) — a verified before/after fix of the documented failure.

---

## Spec Reflection

**One way the spec helped me during implementation:** Writing the Chunking Strategy in
`planning.md` *before* coding forced me to decide that "one chunk ≈ one complete review" was the
right unit, which directly produced the split-on-`\n\n`-first design in `chunk_text()`. Because I'd
already reasoned that reviews are short and self-contained, I knew a naive fixed 400-char split
would slice reviews in half — so I added the 60-char tolerance and word-boundary fallback up front
instead of discovering the problem after seeing bad retrieval. The spec turned a vague "split the
text" task into a concrete, testable target (≤460 chars, no orphan fragments) I could verify by
printing chunks.

**One way my implementation diverged from the spec, and why:** My `planning.md` only specified a
flat "400 characters, 50 overlap" rule. In practice I diverged by adding a **60-character tolerance
band and conditional logic** — most reviews are now kept whole (overlap is *only* applied to the
rare 600+ char review), rather than mechanically character-split with overlap as the plan implied.
I changed this after inspecting real chunks: strict 400-char splitting was producing tiny orphan
fragments (e.g. a 52-char tail) that carried no standalone meaning and would pollute retrieval. The
tolerance keeps each review intact, which matters more for opinion text than hitting an exact
character count.

---

## AI Usage

**Instance 1 — Ingestion & chunking (`ingest.py`)**
- *What I gave the AI:* My `planning.md` Documents and Chunking Strategy sections (400 chars / 50
  overlap, split on `\n\n` first), one example file (Craig Kapp's reviews) showing the `#`-header
  format, and the requirement that each chunk keep `{text, source, professor, rmp_url}` metadata.
- *What it produced:* A `chunk_text()` that did a plain fixed-size character split with overlap on
  the whole document, plus `load_documents()`.
- *What I changed or overrode:* The fixed split was cutting individual reviews into fragments. I
  directed it to **split on review boundaries (`\n\n`) first and keep whole reviews**, and I added
  the **60-char tolerance** so a slightly-over-limit review isn't broken into an orphan. I also
  added `parse_header()`/`clean_text()` to strip the metadata header before chunking, which the
  first version didn't handle.

**Instance 2 — Grounded generation (`query.py`)**
- *What I gave the AI:* My grounding requirement (answer from retrieved context only, cite sources,
  refuse when context is insufficient), the `retrieve()` signature, and the Groq
  `llama-3.3-70b-versatile` target.
- *What it produced:* An `ask()` that built a context prompt and asked the model to "use the
  documents and mention which ones you used."
- *What I changed or overrode:* That left attribution and refusal up to the model. I **tightened the
  system prompt into 5 hard rules** — including the *exact* refusal sentence the model must return —
  and **moved source attribution out of the LLM's hands**: the pipeline now builds the deduplicated
  `sources` list programmatically from chunk metadata, so citations are guaranteed even if the model
  omits them. I also set `temperature=0.2` to reduce drift from the source text.

---

## Stretch Features

Two stretch features are implemented in [`search.py`](search.py) and exposed via the "Search mode"
toggle in the UI (`mode="smart"` in [`ask()`](query.py#L84)). Both target the same weakness the
[failure case](#failure-case-analysis) exposed: when a query is *about* a professor but shares
vocabulary with other professors, pure semantic search ranks the wrong reviews first.

### 1. Metadata Filtering (+1)
[`detect_professor()`](search.py) scans the query for any professor's full name or surname (as a
whole word) and, if found, passes `where={"professor": <name>}` to ChromaDB so retrieval is
restricted to that professor's reviews. The professor name was already stored on every chunk in
[`embed.py`](embed.py#L50). Effect: the query *"What do students say about Craig Kapp's exams and
grading?"* detects `Craig Kapp` and returns **only** Kapp reviews instead of Alan Amin's and Michael
Tao's.

### 2. Hybrid Search — BM25 + Semantic (+2)
[`retrieve_hybrid()`](search.py) combines the two retrieval signals:

```
combined_score = alpha * semantic_similarity_norm + (1 - alpha) * bm25_score_norm   # alpha = 0.5
```

Both components are min-max normalized to [0, 1] over the candidate set so they're comparable.
Semantic similarity comes from ChromaDB (`1 − cosine_distance`); BM25 (`rank_bm25`) scores exact
token overlap, so a surname like "Goldberg" or "Siegel" is weighted heavily instead of being
diluted by topic words. `retrieve_smart()` chains both: detect professor → hybrid search
(filtered if a professor was found).

### Comparison — semantic-only vs. smart (filter + hybrid)

Same three queries, top-5 retrieval, **count of chunks from the correct professor**:

| Query | Semantic-only | Smart (filter + hybrid) |
|-------|:---:|:---:|
| "…Craig Kapp's exams and grading?" | **0 / 5** Kapp (Amin & Tao instead) | **5 / 5** Kapp ✅ |
| "…Ben Goldberg's availability…?" | 2 / 5 Goldberg | **5 / 5** Goldberg ✅ |
| "Is Alan Siegel a good professor for algorithms?" | 4 / 5 Siegel | **5 / 5** Siegel ✅ |

**End-to-end effect on the failure case:** in baseline semantic mode the Kapp question is *refused*
(no Kapp chunks retrieved); in smart mode the system answers correctly — *"tests and quizzes are
fair, with no tricks… clear grading criteria… extra credit available."* (source: craig_kapp.txt.txt).

**Why smart isn't always strictly better:** for queries that *don't* name a professor (e.g. the
"best professor for a beginner" question), `detect_professor()` returns `None`, so smart mode falls
back to unfiltered hybrid search — BM25 contributes little when the query shares no rare tokens with
the reviews, and results are close to semantic-only. The win is concentrated on professor-named
queries, which is the majority of real usage for this domain.

*Reproduce:* `python search.py "What do students say about Craig Kapp's exams and grading?"`
</content>
</invoke>
