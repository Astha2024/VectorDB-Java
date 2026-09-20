# VectorDB — Java Port

A fully working **Vector Database** built from scratch in **Java** with a web UI.
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.

This is a 1:1 port of the original C++ project — same algorithms, same REST API, same frontend. No external Java libraries are used: the HTTP server is `com.sun.net.httpserver.HttpServer` and the Ollama client is `java.net.http.HttpClient`, both built into the JDK.

---

## Prerequisites

You need **2 things** installed (no MSYS2/g++/MinGW needed — Java replaces the compiler toolchain):

1. **JDK 17 or newer** (you're already set — confirmed JDK 21)
2. **Ollama** (runs the local AI models, same as before)

---

## Setup

### Step 1 — Confirm your JDK

```powershell
java --version
javac --version
```

### Step 2 — Install Ollama (unchanged from the C++ version)

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
ollama list
```

### Step 3 — Compile

From the project root:

```powershell
javac -d out src\com\vectordb\*.java
```

This replaces `g++ -std=c++17 -O2 main.cpp -o db -lws2_32`. No linker flags needed — Java's `HttpServer`/`HttpClient` handle networking without a `-lws2_32`-style socket library.

### Step 4 — Run

```powershell
java -cp out com.vectordb.Main
```

You should see the exact same startup banner as the C++ version:

```
=== VectorDB Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open **http://localhost:8080** in your browser — same frontend, unchanged.

---

## Project Structure

```
VectorDB-Java/
├── src/com/vectordb/
│   ├── Main.java          ← HTTP server + route table (was httplib.h usage in main.cpp)
│   ├── VectorItem.java    ← struct VectorItem
│   ├── Distances.java     ← euclidean / cosine / manhattan + DistFn
│   ├── BruteForce.java    ← BruteForce class
│   ├── KDTree.java        ← KDNode / KDTree classes
│   ├── HNSW.java          ← HNSW class (the multilayer graph index)
│   ├── VectorDB.java      ← VectorDB class (16D demo index)
│   ├── DocItem.java       ← struct DocItem
│   ├── DocumentDB.java    ← DocumentDB class (768D Ollama embeddings)
│   ├── OllamaClient.java  ← OllamaClient class (java.net.http.HttpClient)
│   ├── JsonUtil.java      ← jS/jVec/parseVec/extractStr/extractInt helpers
│   ├── TextChunker.java   ← chunkText function
│   └── DemoData.java      ← loadDemo function (the 20 pre-loaded vectors)
├── index.html             ← unchanged frontend
└── README.md
```

## What changed vs. the C++ version

| C++ | Java | Why |
|---|---|---|
| `httplib::Server` | `com.sun.net.httpserver.HttpServer` | Built into the JDK, zero dependencies, mirrors the single-header spirit |
| `httplib::Client` | `java.net.http.HttpClient` | Built into the JDK since Java 11 |
| raw pointers (`KDNode*`) | object references | Java has no manual memory management — GC replaces `destroy()` |
| `std::unordered_map<int,Node>` | `HashMap<Integer,Node>` | Same semantics |
| `std::priority_queue` | `java.util.PriorityQueue` (with `Collections.reverseOrder()` for max-heaps) | Java's PQ is min-heap by default; C++'s is max-heap by default — comparators are flipped where needed |
| `std::mutex` | `ReentrantLock` | Same coarse per-DB locking strategy |
| `std::mt19937 rng(42)` | `java.util.Random(42)` | Same fixed seed, so demo behavior is reproducible (exact HNSW graph shape may still differ slightly since the two PRNGs aren't bit-identical) |
| `g++ -O2` | JIT compilation | No manual optimization flags needed |

All REST endpoints, request/response JSON shapes, and the demo dataset are unchanged — the frontend (`index.html`) works with either backend without modification.


