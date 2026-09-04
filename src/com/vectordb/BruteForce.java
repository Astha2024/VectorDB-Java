package com.vectordb;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import com.vectordb.Distances.DistFn;

/** Direct port of the C++ BruteForce class. O(N*d) exact k-NN, used as ground truth / small-set fallback. */
public class BruteForce {
    public List<VectorItem> items = new ArrayList<>();

    public void insert(VectorItem v) {
        items.add(v);
    }

    /** Returns list of (distance, id) pairs, sorted ascending by distance, truncated to k. */
    public List<Pair> knn(float[] q, int k, DistFn dist) {
        List<Pair> r = new ArrayList<>(items.size());
        for (VectorItem v : items) r.add(new Pair(dist.dist(q, v.emb), v.id));
        r.sort((a, b) -> Float.compare(a.dist, b.dist));
        if (r.size() > k) r = new ArrayList<>(r.subList(0, k));
        return r;
    }

    public void remove(int id) {
        Iterator<VectorItem> it = items.iterator();
        while (it.hasNext()) {
            if (it.next().id == id) it.remove();
        }
    }

    /** Simple (distance, id) pair — stands in for std::pair<float,int>. */
    public static class Pair implements Comparable<Pair> {
        public float dist;
        public int id;

        public Pair(float dist, int id) {
            this.dist = dist;
            this.id = id;
        }

        @Override
        public int compareTo(Pair o) {
            int c = Float.compare(this.dist, o.dist);
            if (c != 0) return c;
            return Integer.compare(this.id, o.id);
        }
    }
}
