package com.vectordb;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.vectordb.Distances.DistFn;

/**
 * Port of the C++ HTTP SERVER section (the main() function's route table).
 * httplib::Server -> com.sun.net.httpserver.HttpServer (built into the JDK,
 * so no external dependency is needed, matching the header-only-library spirit
 * of the original cpp-httplib version).
 */
public class Main {

    static final int DIMS = 16; // demo vectors

    static VectorDB db;
    static DocumentDB docDB;
    static OllamaClient ollama;

    public static void main(String[] args) throws IOException {
        db = new VectorDB(DIMS);
        docDB = new DocumentDB();
        ollama = new OllamaClient();

        db.loadFromDisk();
        if (db.size() == 0) {
            DemoData.load(db);
            db.saveToDisk();
        }
        docDB.loadFromDisk();

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);
        }

        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", 8080), 0);
        server.setExecutor(Executors.newCachedThreadPool());

        server.createContext("/", Main::route);
        server.start();
    }

    // ── Routing ───────────────────────────────────────────────────────────

    private static final Pattern DELETE_ID = Pattern.compile("^/delete/(\\d+)$");
    private static final Pattern DOC_DELETE_ID = Pattern.compile("^/doc/delete/(\\d+)$");

    private static void route(HttpExchange ex) throws IOException {
        try {
            String method = ex.getRequestMethod();
            String path = ex.getRequestURI().getPath();

            if ("OPTIONS".equals(method)) {
                cors(ex);
                ex.sendResponseHeaders(204, -1);
                return;
            }

            if ("GET".equals(method) && "/search".equals(path)) { handleSearch(ex); return; }
            if ("POST".equals(method) && "/insert".equals(path)) { handleInsert(ex); return; }
            if ("GET".equals(method) && "/items".equals(path)) { handleItems(ex); return; }
            if ("GET".equals(method) && "/benchmark".equals(path)) { handleBenchmark(ex); return; }
            if ("GET".equals(method) && "/hnsw-info".equals(path)) { handleHnswInfo(ex); return; }
            if ("GET".equals(method) && "/stats".equals(path)) { handleStats(ex); return; }
            if ("GET".equals(method) && "/status".equals(path)) { handleStatus(ex); return; }

            if ("POST".equals(method) && "/doc/insert".equals(path)) { handleDocInsert(ex); return; }
            if ("GET".equals(method) && "/doc/list".equals(path)) { handleDocList(ex); return; }
            if ("POST".equals(method) && "/doc/search".equals(path)) { handleDocSearch(ex); return; }
            if ("POST".equals(method) && "/doc/ask".equals(path)) { handleDocAsk(ex); return; }

            if ("DELETE".equals(method)) {
                Matcher m1 = DELETE_ID.matcher(path);
                if (m1.matches()) { handleDelete(ex, Integer.parseInt(m1.group(1))); return; }
                Matcher m2 = DOC_DELETE_ID.matcher(path);
                if (m2.matches()) { handleDocDelete(ex, Integer.parseInt(m2.group(1))); return; }
            }

            if ("GET".equals(method) && "/".equals(path)) { handleIndex(ex); return; }

            cors(ex);
            sendJson(ex, 404, "{\"error\":\"not found\"}");
        } catch (Exception e) {
            cors(ex);
            sendJson(ex, 500, "{\"error\":" + JsonUtil.jS(String.valueOf(e.getMessage())) + "}");
        }
    }

    // ── Demo vector endpoints ────────────────────────────────────────────

    private static void handleSearch(HttpExchange ex) throws IOException {
        cors(ex);
        Map<String, String> params = queryParams(ex.getRequestURI());
        float[] q = JsonUtil.parseVec(params.getOrDefault("v", ""));
        if (q.length != DIMS) {
            sendJson(ex, 200, "{\"error\":\"need " + DIMS + "D vector\"}");
            return;
        }
        int k = parseIntOr(params.get("k"), 5);
        String metric = params.getOrDefault("metric", "cosine");
        if (metric.isEmpty()) metric = "cosine";
        String algo = params.getOrDefault("algo", "hnsw");
        if (algo.isEmpty()) algo = "hnsw";

        VectorDB.SearchOut out = db.search(q, k, metric, algo);
        StringBuilder ss = new StringBuilder();
        ss.append("{\"results\":[");
        for (int i = 0; i < out.hits.size(); i++) {
            if (i > 0) ss.append(',');
            VectorDB.Hit h = out.hits.get(i);
            ss.append("{\"id\":").append(h.id)
              .append(",\"metadata\":").append(JsonUtil.jS(h.meta))
              .append(",\"category\":").append(JsonUtil.jS(h.cat))
              .append(",\"distance\":").append(String.format(java.util.Locale.ROOT, "%.6f", h.dist))
              .append(",\"embedding\":").append(JsonUtil.jVec(h.emb)).append('}');
        }
        ss.append("],\"latencyUs\":").append(out.us)
          .append(",\"algo\":").append(JsonUtil.jS(out.algo))
          .append(",\"metric\":").append(JsonUtil.jS(out.metric)).append('}');
        sendJson(ex, 200, ss.toString());
    }

    private static void handleInsert(HttpExchange ex) throws IOException {
        cors(ex);
        String body = readBody(ex);
        JsonUtil.InsertBody parsed = JsonUtil.parseInsertBody(body);
        if (!parsed.ok || parsed.emb.length != DIMS) {
            sendJson(ex, 200, "{\"error\":\"invalid body\"}");
            return;
        }
        DistFn dist = Distances.getDistFn("cosine");
        int id = db.insert(parsed.meta, parsed.cat, parsed.emb, dist);
        db.saveToDisk();
        sendJson(ex, 200, "{\"id\":" + id + "}");
    }

    private static void handleDelete(HttpExchange ex, int id) throws IOException {
        cors(ex);
        boolean ok = db.remove(id);
        if (ok) db.saveToDisk();
        sendJson(ex, 200, "{\"ok\":" + ok + "}");
    }

    private static void handleItems(HttpExchange ex) throws IOException {
        cors(ex);
        List<VectorItem> items = db.all();
        StringBuilder ss = new StringBuilder("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) ss.append(',');
            VectorItem v = items.get(i);
            ss.append("{\"id\":").append(v.id)
              .append(",\"metadata\":").append(JsonUtil.jS(v.metadata))
              .append(",\"category\":").append(JsonUtil.jS(v.category))
              .append(",\"embedding\":").append(JsonUtil.jVec(v.emb)).append('}');
        }
        ss.append(']');
        sendJson(ex, 200, ss.toString());
    }

    private static void handleBenchmark(HttpExchange ex) throws IOException {
        cors(ex);
        Map<String, String> params = queryParams(ex.getRequestURI());
        float[] q = JsonUtil.parseVec(params.getOrDefault("v", ""));
        if (q.length != DIMS) {
            sendJson(ex, 200, "{\"error\":\"need " + DIMS + "D vector\"}");
            return;
        }
        int k = parseIntOr(params.get("k"), 5);
        String metric = params.getOrDefault("metric", "cosine");
        if (metric.isEmpty()) metric = "cosine";

        VectorDB.BenchOut b = db.benchmark(q, k, metric);
        String json = "{\"bruteforceUs\":" + b.bfUs + ",\"kdtreeUs\":" + b.kdUs
                + ",\"hnswUs\":" + b.hnswUs + ",\"itemCount\":" + b.n + "}";
        sendJson(ex, 200, json);
    }

    private static void handleHnswInfo(HttpExchange ex) throws IOException {
        cors(ex);
        HNSW.GraphInfo gi = db.hnswInfo();
        StringBuilder ss = new StringBuilder();
        ss.append("{\"topLayer\":").append(gi.topLayer)
          .append(",\"nodeCount\":").append(gi.nodeCount)
          .append(",\"nodesPerLayer\":[");
        for (int i = 0; i < gi.nodesPerLayer.size(); i++) {
            if (i > 0) ss.append(',');
            ss.append(gi.nodesPerLayer.get(i));
        }
        ss.append("],\"edgesPerLayer\":[");
        for (int i = 0; i < gi.edgesPerLayer.size(); i++) {
            if (i > 0) ss.append(',');
            ss.append(gi.edgesPerLayer.get(i));
        }
        ss.append("],\"nodes\":[");
        for (int i = 0; i < gi.nodes.size(); i++) {
            if (i > 0) ss.append(',');
            HNSW.NodeView n = gi.nodes.get(i);
            ss.append("{\"id\":").append(n.id)
              .append(",\"metadata\":").append(JsonUtil.jS(n.metadata))
              .append(",\"category\":").append(JsonUtil.jS(n.category))
              .append(",\"maxLyr\":").append(n.maxLyr).append('}');
        }
        ss.append("],\"edges\":[");
        for (int i = 0; i < gi.edges.size(); i++) {
            if (i > 0) ss.append(',');
            HNSW.EdgeView e = gi.edges.get(i);
            ss.append("{\"src\":").append(e.src).append(",\"dst\":").append(e.dst)
              .append(",\"lyr\":").append(e.lyr).append('}');
        }
        ss.append("]}");
        sendJson(ex, 200, ss.toString());
    }

    private static void handleStats(HttpExchange ex) throws IOException {
        cors(ex);
        String json = "{\"count\":" + db.size()
                + ",\"dims\":" + DIMS
                + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
                + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
        sendJson(ex, 200, json);
    }

    // ── Document + RAG endpoints ─────────────────────────────────────────

    private static void handleDocInsert(HttpExchange ex) throws IOException {
        cors(ex);
        String body = readBody(ex);
        String title = JsonUtil.extractStr(body, "title");
        String text = JsonUtil.extractStr(body, "text");
        if (title.isEmpty() || text.isEmpty()) {
            sendJson(ex, 200, "{\"error\":\"need title and text\"}");
            return;
        }

        List<String> chunks = TextChunker.chunkText(text, 250, 30);
        StringBuilder ids = new StringBuilder();
        for (int i = 0; i < chunks.size(); i++) {
            float[] emb = ollama.embed(chunks.get(i));
            if (emb.length == 0) {
                sendJson(ex, 200,
                        "{\"error\":\"Ollama unavailable. "
                                + "Install from https://ollama.com then run: "
                                + "ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                return;
            }
            String chunkTitle = (chunks.size() > 1)
                    ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                    : title;
            int id = docDB.insert(chunkTitle, chunks.get(i), emb);
            if (i > 0) ids.append(',');
            ids.append(id);
        }

        String json = "{\"ids\":[" + ids + "],\"chunks\":" + chunks.size()
                + ",\"dims\":" + docDB.getDims() + "}";
        docDB.saveToDisk();
        sendJson(ex, 200, json);
    }

    private static void handleDocDelete(HttpExchange ex, int id) throws IOException {
        cors(ex);
        boolean ok = docDB.remove(id);
        if (ok) docDB.saveToDisk();
        sendJson(ex, 200, "{\"ok\":" + ok + "}");
    }

    private static void handleDocList(HttpExchange ex) throws IOException {
        cors(ex);
        List<DocItem> docs = docDB.all();
        StringBuilder ss = new StringBuilder("[");
        for (int i = 0; i < docs.size(); i++) {
            if (i > 0) ss.append(',');
            DocItem d = docs.get(i);
            String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "\u2026" : d.text;
            int words = 1;
            for (char c : d.text.toCharArray()) if (c == ' ') words++;
            ss.append("{\"id\":").append(d.id)
              .append(",\"title\":").append(JsonUtil.jS(d.title))
              .append(",\"preview\":").append(JsonUtil.jS(preview))
              .append(",\"words\":").append(words).append('}');
        }
        ss.append(']');
        sendJson(ex, 200, ss.toString());
    }

    private static void handleDocSearch(HttpExchange ex) throws IOException {
        cors(ex);
        String body = readBody(ex);
        String question = JsonUtil.extractStr(body, "question");
        int k = JsonUtil.extractInt(body, "k", 3);
        if (question.isEmpty()) {
            sendJson(ex, 200, "{\"error\":\"need question\"}");
            return;
        }
        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) {
            sendJson(ex, 200, "{\"error\":\"Ollama unavailable\"}");
            return;
        }
        List<DocumentDB.ScoredDoc> hits = docDB.search(qEmb, k);
        StringBuilder ss = new StringBuilder("{\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) ss.append(',');
            DocumentDB.ScoredDoc h = hits.get(i);
            ss.append("{\"id\":").append(h.doc.id)
              .append(",\"title\":").append(JsonUtil.jS(h.doc.title))
              .append(",\"distance\":").append(String.format(java.util.Locale.ROOT, "%.4f", h.dist)).append('}');
        }
        ss.append("]}");
        sendJson(ex, 200, ss.toString());
    }

    private static void handleDocAsk(HttpExchange ex) throws IOException {
        cors(ex);
        String body = readBody(ex);
        String question = JsonUtil.extractStr(body, "question");
        int k = JsonUtil.extractInt(body, "k", 3);
        if (question.isEmpty()) {
            sendJson(ex, 200, "{\"error\":\"need question\"}");
            return;
        }

        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) {
            sendJson(ex, 200, "{\"error\":\"Ollama unavailable\"}");
            return;
        }

        List<DocumentDB.ScoredDoc> hits = docDB.search(qEmb, k);

        StringBuilder ctx = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            DocumentDB.ScoredDoc h = hits.get(i);
            ctx.append('[').append(i + 1).append("] ").append(h.doc.title).append(":\n")
               .append(h.doc.text).append("\n\n");
        }

        String prompt = "You are a helpful assistant. Answer the user's question directly. "
                + "Use the provided context if it contains relevant information. "
                + "If it doesn't, just use your own general knowledge. "
                + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                + "Just answer the question naturally.\n\n"
                + "Context:\n" + ctx
                + "Question: " + question + "\n\n"
                + "Answer:";

        String answer = ollama.generate(prompt);

        StringBuilder ss = new StringBuilder();
        ss.append("{\"answer\":").append(JsonUtil.jS(answer))
          .append(",\"model\":").append(JsonUtil.jS(ollama.genModel))
          .append(",\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) ss.append(',');
            DocumentDB.ScoredDoc h = hits.get(i);
            ss.append("{\"id\":").append(h.doc.id)
              .append(",\"title\":").append(JsonUtil.jS(h.doc.title))
              .append(",\"text\":").append(JsonUtil.jS(h.doc.text))
              .append(",\"distance\":").append(String.format(java.util.Locale.ROOT, "%.4f", h.dist)).append('}');
        }
        ss.append("],\"docCount\":").append(docDB.size()).append('}');
        sendJson(ex, 200, ss.toString());
    }

    private static void handleStatus(HttpExchange ex) throws IOException {
        cors(ex);
        boolean up = ollama.isAvailable();
        String json = "{\"ollamaAvailable\":" + up
                + ",\"embedModel\":" + JsonUtil.jS(ollama.embedModel)
                + ",\"genModel\":" + JsonUtil.jS(ollama.genModel)
                + ",\"docCount\":" + docDB.size()
                + ",\"docDims\":" + docDB.getDims()
                + ",\"demoDims\":" + DIMS
                + ",\"demoCount\":" + db.size() + "}";
        sendJson(ex, 200, json);
    }

    // ── Static file (index.html) ─────────────────────────────────────────

    private static void handleIndex(HttpExchange ex) throws IOException {
        Path p = Path.of("index.html");
        if (!Files.exists(p)) {
            ex.sendResponseHeaders(404, -1);
            return;
        }
        byte[] content = Files.readAllBytes(p);
        ex.getResponseHeaders().add("Content-Type", "text/html; charset=utf-8");
        ex.sendResponseHeaders(200, content.length);
        try (var os = ex.getResponseBody()) {
            os.write(content);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static void cors(HttpExchange ex) {
        ex.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        ex.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");
    }

    private static void sendJson(HttpExchange ex, int status, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json");
        ex.sendResponseHeaders(status, bytes.length);
        try (var os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static String readBody(HttpExchange ex) throws IOException {
        try (InputStream is = ex.getRequestBody()) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            int n;
            while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
            return bos.toString(StandardCharsets.UTF_8);
        }
    }

    private static Map<String, String> queryParams(URI uri) {
        Map<String, String> map = new java.util.HashMap<>();
        String query = uri.getRawQuery();
        if (query == null || query.isEmpty()) return map;
        for (String pair : query.split("&")) {
            int idx = pair.indexOf('=');
            if (idx < 0) continue;
            String key = java.net.URLDecoder.decode(pair.substring(0, idx), StandardCharsets.UTF_8);
            String val = java.net.URLDecoder.decode(pair.substring(idx + 1), StandardCharsets.UTF_8);
            map.put(key, val);
        }
        return map;
    }

    private static int parseIntOr(String s, int def) {
        if (s == null) return def;
        try {
            return Integer.parseInt(s);
        } catch (NumberFormatException e) {
            return def;
        }
    }
}
