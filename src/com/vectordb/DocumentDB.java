package com.vectordb;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import com.vectordb.BruteForce.Pair;

/** Port of the C++ DocumentDB class: HNSW over real Ollama embeddings, with a brute-force fallback for small sets. */
public class DocumentDB {
    private final Map<Integer, DocItem> store = new HashMap<>();
    private final HNSW hnsw = new HNSW(16, 200);
    private final BruteForce bf = new BruteForce();
    private final ReentrantLock lock = new ReentrantLock();
    private int nextId = 1;
    private int dims = 0; // determined from the first inserted embedding

    public int insert(String title, String text, float[] emb) {
        lock.lock();
        try {
            if (dims == 0) dims = emb.length;
            DocItem item = new DocItem(nextId++, title, text, emb);
            store.put(item.id, item);
            VectorItem vi = new VectorItem(item.id, title, "doc", emb);
            hnsw.insert(vi, Distances::cosine);
            bf.insert(vi);
            return item.id;
        } finally {
            lock.unlock();
        }
    }

    public static class ScoredDoc {
        public float dist;
        public DocItem doc;

        public ScoredDoc(float dist, DocItem doc) {
            this.dist = dist;
            this.doc = doc;
        }
    }

    /** Semantic search: top-k most similar chunks within max_dist (cosine distance). */
    public List<ScoredDoc> search(float[] q, int k, float maxDist) {
        lock.lock();
        try {
            if (store.isEmpty()) return new ArrayList<>();
            List<Pair> raw = (store.size() < 10)
                    ? bf.knn(q, k, Distances::cosine)
                    : hnsw.knn(q, k, 50, Distances::cosine);
            List<ScoredDoc> out = new ArrayList<>();
            for (Pair p : raw) {
                DocItem d = store.get(p.id);
                if (d != null && p.dist <= maxDist) out.add(new ScoredDoc(p.dist, d));
            }
            return out;
        } finally {
            lock.unlock();
        }
    }

    public List<ScoredDoc> search(float[] q, int k) {
        return search(q, k, 0.7f);
    }

    public boolean remove(int id) {
        lock.lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            hnsw.remove(id);
            bf.remove(id);
            return true;
        } finally {
            lock.unlock();
        }
    }

    public List<DocItem> all() {
        lock.lock();
        try {
            return new ArrayList<>(store.values());
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

    public int getDims() {
        return dims;
    }
}
