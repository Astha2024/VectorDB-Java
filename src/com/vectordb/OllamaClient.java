package com.vectordb;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Port of the C++ OllamaClient class. httplib::Client is replaced by java.net.http.HttpClient
 * (built into the JDK since Java 11 — no external dependency needed, matching the header-only
 * spirit of the original).
 */
public class OllamaClient {
    private final String host;
    private final int port;
    public String embedModel = "nomic-embed-text";
    public String genModel = "llama3.2";

    public OllamaClient() {
        this("127.0.0.1", 11434);
    }

    public OllamaClient(String host, int port) {
        this.host = host;
        this.port = port;
    }

    private String baseUrl() {
        return "http://" + host + ":" + port;
    }

    /** Escape a string for embedding inside a JSON string literal. */
    private String esc(String s) {
        StringBuilder o = new StringBuilder();
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"': o.append("\\\""); break;
                case '\\': o.append("\\\\"); break;
                case '\n': o.append("\\n"); break;
                case '\r': o.append("\\r"); break;
                case '\t': o.append("\\t"); break;
                default: o.append(c);
            }
        }
        return o.toString();
    }

    public boolean isAvailable() {
        try {
            HttpClient cli = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(2))
                    .build();
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl() + "/api/tags"))
                    .GET()
                    .build();
            HttpResponse<String> res = cli.send(req, HttpResponse.BodyHandlers.ofString());
            return res.statusCode() == 200;
        } catch (IOException | InterruptedException e) {
            return false;
        }
    }

    /** Returns an empty array if Ollama is not running or the model isn't found. */
    public float[] embed(String text) {
        try {
            HttpClient cli = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(3))
                    .build();
            String body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl() + "/api/embeddings"))
                    .timeout(Duration.ofSeconds(30))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> res = cli.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) return new float[0];
            return parseEmbedding(res.body());
        } catch (IOException | InterruptedException e) {
            return new float[0];
        }
    }

    /** Returns an error string if Ollama is unavailable, matching the C++ behavior. */
    public String generate(String prompt) {
        try {
            HttpClient cli = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(3))
                    .build();
            String body = "{\"model\":\"" + genModel + "\","
                    + "\"prompt\":\"" + esc(prompt) + "\","
                    + "\"stream\":false}";
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl() + "/api/generate"))
                    .timeout(Duration.ofSeconds(180)) // LLMs can be slow
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> res = cli.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) return "ERROR: Ollama unavailable. Run: ollama serve";
            return JsonUtil.extractStr(res.body(), "response");
        } catch (IOException | InterruptedException e) {
            return "ERROR: Ollama unavailable. Run: ollama serve";
        }
    }

    /** Parse {"embedding":[...]} from Ollama's /api/embeddings response, handling nested brackets. */
    private float[] parseEmbedding(String body) {
        int p = body.indexOf("\"embedding\"");
        if (p < 0) return new float[0];
        p = body.indexOf('[', p);
        if (p < 0) return new float[0];
        int e = p + 1, depth = 1;
        while (e < body.length() && depth > 0) {
            char c = body.charAt(e);
            if (c == '[') depth++;
            else if (c == ']') depth--;
            e++;
        }
        return JsonUtil.parseVec(body.substring(p + 1, e - 1));
    }
}
