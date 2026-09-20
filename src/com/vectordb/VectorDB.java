package com.vectordb;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import com.vectordb.Distances.DistFn;
import com.vectordb.BruteForce.Pair;

/**
 * Port of the C++ VectorDB class: unified interface over BruteForce, KDTree, and HNSW
 * for the 16D demo vectors. std::mutex -> ReentrantLock.
 */
public class VectorDB {
    private final Map<Integer, VectorItem> store = new HashMap<>();
    private final BruteForce bf = new BruteForce();
    private final KDTree kdt;
    private final HNSW hnsw = new HNSW(16, 200);
    private final ReentrantLock lock = new ReentrantLock();
    private int nextId = 1;
    public final int dims;

    public VectorDB(int dims) {
        this.dims = dims;
        this.kdt = new KDTree(dims);
    }

    public int insert(String meta, String cat, float[] emb, DistFn dist) {
        lock.lock();
        try {
            VectorItem v = new VectorItem(nextId++, meta, cat, emb);
            store.put(v.id, v);
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            return v.id;
        } finally {
            lock.unlock();
        }
    }

    public boolean remove(int id) {
        lock.lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            bf.remove(id);
            hnsw.remove(id);
            kdt.rebuild(new ArrayList<>(store.values()));
            return true;
        } finally {
            lock.unlock();
        }
    }

    public static class Hit {
        public int id;
        public String meta, cat;
        public float[] emb;
        public float dist;

        public Hit(int id, String meta, String cat, float[] emb, float dist) {
            this.id = id;
            this.meta = meta;
            this.cat = cat;
            this.emb = emb;
            this.dist = dist;
        }
    }

    public static class SearchOut {
        public List<Hit> hits = new ArrayList<>();
        public long us;
        public String algo, metric;
    }

    public SearchOut search(float[] q, int k, String metric, String algo) {
        lock.lock();
        try {
            DistFn dfn = Distances.getDistFn(metric);
            long t0 = System.nanoTime();
            List<Pair> raw;
            if ("bruteforce".equals(algo)) raw = bf.knn(q, k, dfn);
            else if ("kdtree".equals(algo)) raw = kdt.knn(q, k, dfn);
            else raw = hnsw.knn(q, k, 50, dfn);
            long us = (System.nanoTime() - t0) / 1000;

            SearchOut out = new SearchOut();
            out.us = us;
            out.algo = algo;
            out.metric = metric;
            for (Pair p : raw) {
                VectorItem v = store.get(p.id);
                if (v != null) out.hits.add(new Hit(p.id, v.metadata, v.category, v.emb, p.dist));
            }
            return out;
        } finally {
            lock.unlock();
        }
    }

    public static class BenchOut {
        public long bfUs, kdUs, hnswUs;
        public int n;
    }

    public BenchOut benchmark(float[] q, int k, String metric) {
        lock.lock();
        try {
            DistFn dfn = Distances.getDistFn(metric);
            BenchOut b = new BenchOut();

            long t0 = System.nanoTime();
            bf.knn(q, k, dfn);
            b.bfUs = (System.nanoTime() - t0) / 1000;

            t0 = System.nanoTime();
            kdt.knn(q, k, dfn);
            b.kdUs = (System.nanoTime() - t0) / 1000;

            t0 = System.nanoTime();
            hnsw.knn(q, k, 50, dfn);
            b.hnswUs = (System.nanoTime() - t0) / 1000;

            b.n = store.size();
            return b;
        } finally {
            lock.unlock();
        }
    }

    public List<VectorItem> all() {
        lock.lock();
        try {
            return new ArrayList<>(store.values());
        } finally {
            lock.unlock();
        }
    }

    public HNSW.GraphInfo hnswInfo() {
        lock.lock();
        try {
            return hnsw.getInfo();
        } finally {
            lock.unlock();
        }
    }

    public int size() {
        lock.lock();
        try {
            return store.size();
        } finally {
            lock.unlock();
        }
    }

    public void saveToDisk() {
        lock.lock();
        try {
            java.util.List<String> lines = new ArrayList<>();
            for (VectorItem d : store.values()) {
                String line = "{\"metadata\":" + JsonUtil.jS(d.metadata) + 
                              ",\"category\":" + JsonUtil.jS(d.category) + 
                              ",\"embedding\":" + JsonUtil.jVec(d.emb) + "}";
                lines.add(line);
            }
            java.nio.file.Files.write(java.nio.file.Path.of("vectors.jsonl"), lines, java.nio.charset.StandardCharsets.UTF_8);
        } catch (Exception e) {
            System.err.println("Failed to save vectors: " + e.getMessage());
        } finally {
            lock.unlock();
        }
    }

    public void loadFromDisk() {
        lock.lock();
        try {
            java.nio.file.Path p = java.nio.file.Path.of("vectors.jsonl");
            if (!java.nio.file.Files.exists(p)) return;
            java.util.List<String> lines = java.nio.file.Files.readAllLines(p, java.nio.charset.StandardCharsets.UTF_8);
            DistFn dist = Distances.getDistFn("cosine");
            for (String line : lines) {
                if (line.trim().isEmpty()) continue;
                String meta = JsonUtil.extractStr(line, "metadata");
                String cat = JsonUtil.extractStr(line, "category");
                float[] emb = JsonUtil.extractFloatArray(line, "embedding");
                if (!meta.isEmpty() && emb.length > 0) {
                    this.insert(meta, cat, emb, dist);
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to load vectors: " + e.getMessage());
        } finally {
            lock.unlock();
        }
    }
}
