# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

Student reviews of computer science professors at New York University (NYU), sourced from Rate My Professors (ratemyprofessors.com). This knowledge is valuable because it captures real student experiences( exam difficulty, grading style, how helpful a professor is outside class, and overall workload)none of which appear in official course catalogs or the NYU website. A student deciding between two professors for the same CS course has no official way to compare them; this system makes that search instant and cited.

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

10 NYU CS professors as sources.

| #   | Source            | Description                                                                                                                 | URL or location                                    |
| --- | ----------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| 1   | Craig Kapp        | Intro CS (CSCI-UA 2) — 159 reviews, 4.8/5. Known for amazing lectures, fair tests, extra credit, and being inspirational.   | https://www.ratemyprofessors.com/professor/1579749 |
| 2   | Joshua Clayton    | CS (CSCI-UA 4) — 67 reviews, 4.7/5. Highly rated for clear teaching and accessibility.                                      | https://www.ratemyprofessors.com/professor/1808953 |
| 3   | Ben Goldberg      | Data Structures (CSCI-UA 201) — 65 reviews, 4.7/5. Strong reviews for clarity and feedback.                                 | https://www.ratemyprofessors.com/professor/216065  |
| 4   | Anirudh Sivaraman | Networking (CS 480) — 18 reviews, 4.6/5. Upper-level course, positive ratings.                                              | https://www.ratemyprofessors.com/professor/2329940 |
| 5   | Evan Korth        | CS102 — 114 reviews, 3.5/5. Polarizing: some love, some find tough.                                                         | https://www.ratemyprofessors.com/professor/272737  |
| 6   | Alan Siegel       | Algorithms (CSCI 310) — 69 reviews, 2.7/5. Very low rating; known as a tough, unclear grader.                               | https://www.ratemyprofessors.com/professor/1941844 |
| 7   | Michael Tao       | Intro CS (CSCI-UA 101) — 18 reviews, 2.8/5. Extremely long, hard exams; half the class fails midterms. Nice but test-heavy. | https://www.ratemyprofessors.com/professor/2910630 |
| 8   | Candido Cabo      | Intro CS (CSCI-UA 101) — 17 reviews, mixed (~4.0 effective). Boring lectures but generous curves on tricky exams.           | https://www.ratemyprofessors.com/professor/2567373 |
| 9   | Patrick Cousot    | PL / Static Analysis (CSCI 3140) — 20 reviews, 5.0/5. Inspirational, dense material, exams dig deep into specific topics.   | https://www.ratemyprofessors.com/professor/1762046 |
| 10  | Alan Amin         | CS102 — 7 reviews, 3.9/5. Weekly quizzes and HWs, accessible office hours, hard but lenient grading.                        | https://www.ratemyprofessors.com/professor/2956463 |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:400 characters (~60-80 words)**

**Overlap:50 characters**

\*\*Reasoning:Each Rate My Professor page contains multiple individual reviews (5-20 reviews per page). Each Individual review tends to be short, about 2 sentences. A chunk at 400 means most chucks will be one review.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:all-MiniLM-L6-v2 via sentence-transformers**

**Top-k:5 each chunk is about one student review so k=5 gives the LLM 5 different student perspectives which is enough context**

**Production tradeoff reflection:**

If cost wasn't an issue, I'd probably use OpenAI's text-embedding-3-small. It has an 8192 token context window, which means I could keep entire reviews intact without chunking. It also supports multiple languages, which helps for NYU since sometimes international students leave reviews in other languages. The main downside is needing an API key and paying per token.

For a local upgrade path, bge-large-en-v1.5 scores higher than MiniLM on benchmarks and still runs locally, but it's slower and uses ~1.5GB of RAM. If I were expanding beyond NYU to campuses with more international students, multilingual-e5-large would be the best choice for non-English reviews.

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| #   | Question                                                                     | Expected answer                                                                                                                                                                                                                                   |
| --- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | What do students say about Craig Kapp's exams and grading?                   | Students say his tests and quizzes are fair with no tricks, there's extra credit, and it's easy to get a good grade if you do the work. Difficulty is rated around 2-3/5.                                                                         |
| 2   | What is the main complaint students have about Michael Tao's intro CS class? | The exams are extremely long, confusing, and much harder than the lectures. Students report not finishing exams and that half the class fails the second midterm.                                                                                 |
| 3   | Is Alan Siegel a good professor for algorithms?                              | No he has a 2.7/5 rating with only 35% would take again. Students describe him as a tough grader with test-heavy courses, though some say he's caring but the class is very hard.                                                                 |
| 4   | What do students say about Ben Goldberg's availability and responsiveness?   | responds quickly, very nice about extensions, accessible outside class; 4.7/5 overall                                                                                                                                                             |
| 5   | Which professor would be best for a beginner with no programming experience? | Craig Kapp or Candido Cabo. Kapp is rated 4.8/5 with reviews saying "gave me such a good intro to the computer programming world." Cabo is described as "great at explaining concepts to beginners" with manageable homework and generous curves. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.not enough reviews for certain professors so there isn't enough information to answer confidently.
To fix it, document this as a known limitation; the system should say "I don't have enough information" rather than generate a plausible-sounding answer from thin evidence.

2.the students might refer to the professor by different names like last name only, first name, or course number. A search might not connect the query to a review because of the nickname.
To fix it, include the professor's full name in the header metadata for every chunk, so retrieval can fall back on metadata filtering if needed.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

[1. Document Ingestion]
Raw .txt files in docs/ folder (manually copied from RMP)
Tool: Python open() / os.listdir()
│
▼
[2. Cleaning]
Remove: # header comment lines at top of each file
Keep: rating line, course tag, date, full review text
Tool: Python string methods in ingest.py
│
▼
[3. Chunking]
Split on \n\n (review boundaries) → max 400 chars + 60 char tolerance
Word-boundary splitting with 50 char overlap for long reviews
Attach metadata: {source_file, professor_name, rmp_url}
Tool: Custom chunk_text() in ingest.py
│
▼
[4. Embedding + Vector Store]
Embed each chunk → 384-dimensional vector
Store vectors + metadata in local ChromaDB collection
Tool: sentence-transformers (all-MiniLM-L6-v2) + ChromaDB
│
▼
[5. Retrieval]
User query → embed query → cosine similarity search → top-5 chunks
Return chunks with source metadata (professor name, filename)
Tool: ChromaDB .query()
│
▼
[6. Grounded Generation]
Retrieved chunks → grounding prompt (answer from context only) → LLM
Output: answer text + list of source documents cited
Tool: Groq API (llama-3.3-70b-versatile)
│
▼
[7. Query Interface]
Text input box → answer display + sources display
Tool: Gradio (gr.Blocks) in app.py

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

## AI Tool Plan

Below is my plan for using AI to generate each pipeline component:
**Milestone 3 — Ingestion and chunking:**

- **AI Tool:** Claude
- **What I'll give it:** The "Documents" section from planning.md, the "Chunking Strategy" section (400 characters, 50 overlap, split on `\n\n` first), one example `.txt` file content (Craig Kapp's file), and the requirement that each chunk must retain metadata: `{text, source_file, professor_name}`
- **What I expect it to produce:** `ingest.py` containing: `load_documents(folder_path)` that reads all `.txt` files in `docs/`; `chunk_text(text, chunk_size=400, overlap=50)` that splits on `\n\n` then applies character-based chunking; main function that returns a list of chunks as dictionaries with metadata attached
- **How I'll verify:** Run `python ingest.py` and print first 5 chunks. Manually check each chunk is ≤ 400 characters. Check that no chunk splits a review in half. Check metadata correctly shows filename and professor name for each chunk.

**Milestone 4 — Embedding and retrieval:**

- **AI Tool:** Claude
- **What I'll give it:** The "Retrieval Approach" section (`all-MiniLM-L6-v2`, k=5), the pipeline Architecture diagram, and the output structure of `load_documents()` from Milestone 3 (list of `{text, professor, source, rmp_url}` dicts)
- **What I expect it to produce:** `embed.py` that loads chunks via `get_chunks()`, embeds them with `SentenceTransformer("all-MiniLM-L6-v2")`, and stores them in a local ChromaDB collection with source metadata attached. Also a `retrieve(query, k=5)` function in `query.py` that returns the top-k chunks with their source filenames and cosine distance scores
- **How I'll verify:** Run 3 of my 5 evaluation questions through `retrieve()`, print the returned chunks and distance scores. Confirm chunks are topically relevant and that distance scores are below 0.5 for the top result.

**Milestone 5 — Generation and interface:**

- **AI Tool:** Claude
- **What I'll give it:** The grounding requirement (LLM must answer from retrieved context only, must cite source filenames, must say "I don't have enough information" if context is insufficient), the `retrieve()` function signature, and the Gradio skeleton from the project instructions
- **What I expect it to produce:** An `ask(question)` function in `query.py` that builds a grounding prompt, calls the Groq API (`llama-3.3-70b-versatile`), and returns `{answer, sources}`. A complete `app.py` with a text input field, an answer output box, and a sources output box
- **How I'll verify:** Ask one question covered by the documents and confirm the response cites a specific source file. Then ask one question not in any document and confirm the system says it doesn't have enough information rather than generating a plausible-sounding answer.


---

## Stretch Features (added after the base system worked)

My evaluation surfaced a clear failure: a question *about* one professor ("What do students say about Craig Kapp's exams and grading?") retrieved Alan Amin's and Michael Tao's reviews instead of Kapp's, because the topic words ("exams", "grading") dominated the embedding over the name "Kapp". Both stretch features below target that failure. Implemented in `search.py`.

**Metadata Filtering (+1):**
- **Plan:** Detect a professor name (full name or surname as a whole word) in the query, then restrict ChromaDB retrieval with `where={"professor": <name>}`. The professor name is already attached to every chunk's metadata in `embed.py`, so no re-ingestion is needed.
- **Expected effect:** A query naming "Kapp" returns only Kapp reviews.
- **Verify:** Re-run the failed Kapp query and confirm 5/5 retrieved chunks are now Craig Kapp.

**Hybrid Search -- BM25 + semantic (+2):**
- **Plan:** Combine normalized semantic similarity (ChromaDB cosine) with normalized BM25 keyword scores (`rank_bm25`): `combined = 0.5 * sem_norm + 0.5 * bm25_norm`. BM25 weights the exact surname token so the name is no longer washed out.
- **Expected effect:** Professor-named queries rank that professor's reviews first even without filtering; filtering + hybrid together give the strongest results.
- **Verify:** Compare semantic-only vs. smart (filter + hybrid) on 3 queries and count correct-professor chunks in the top 5 (reported in README).

**AI Tool Plan for stretch:** I gave Claude my failure-case analysis and the two strategies above and asked it to implement `detect_professor()`, `retrieve_semantic(..., professor=...)`, and `retrieve_hybrid(..., alpha=...)` in a new `search.py`, plus a `mode` toggle in `ask()`/`app.py`. I verified by re-running the Kapp query end-to-end in both modes and confirming the before/after.
