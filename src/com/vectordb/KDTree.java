package com.vectordb;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Collections;

import com.vectordb.Distances.DistFn;
import com.vectordb.BruteForce.Pair;

/** Port of the C++ KDNode / KDTree classes. Java references replace raw pointers; no manual destroy() needed (GC). */
public class KDTree {

    private static class KDNode {
        VectorItem item;
        KDNode left, right;

        KDNode(VectorItem v) {
            this.item = v;
        }
    }

    private KDNode root;
    private final int dims;

    public KDTree(int dims) {
        this.dims = dims;
    }

    private KDNode ins(KDNode n, VectorItem v, int d) {
        if (n == null) return new KDNode(v);
        int ax = d % dims;
        if (v.emb[ax] < n.item.emb[ax]) n.left = ins(n.left, v, d + 1);
        else n.right = ins(n.right, v, d + 1);
        return n;
    }

    public void insert(VectorItem v) {
        root = ins(root, v, 0);
    }

    // Java's PriorityQueue is a min-heap; the C++ code uses a max-heap (std::priority_queue<pair<float,int>>)
    // to keep the *worst* of the current top-k at the top so it can be evicted. We replicate that by
    // reversing the comparator (max-heap on distance).
    private void knnRec(KDNode n, float[] q, int k, int d, DistFn dist, PriorityQueue<Pair> heap) {
        if (n == null) return;
        float dn = dist.dist(q, n.item.emb);
        if (heap.size() < k || dn < heap.peek().dist) {
            heap.add(new Pair(dn, n.item.id));
            if (heap.size() > k) heap.poll();
        }
        int ax = d % dims;
        float diff = q[ax] - n.item.emb[ax];
        KDNode closer = diff < 0 ? n.left : n.right;
        KDNode farther = diff < 0 ? n.right : n.left;
        knnRec(closer, q, k, d + 1, dist, heap);
        if (heap.size() < k || Math.abs(diff) < heap.peek().dist) {
            knnRec(farther, q, k, d + 1, dist, heap);
        }
    }

    public List<Pair> knn(float[] q, int k, DistFn dist) {
        // Max-heap: largest distance at the head, so we can evict the worst candidate.
        PriorityQueue<Pair> heap = new PriorityQueue<>(Collections.reverseOrder());
        knnRec(root, q, k, 0, dist, heap);
        List<Pair> r = new ArrayList<>(heap);
        r.sort((a, b) -> Float.compare(a.dist, b.dist));
        return r;
    }

    public void rebuild(List<VectorItem> items) {
        root = null; // old tree becomes unreachable and is garbage-collected (no manual destroy needed)
        for (VectorItem v : items) insert(v);
    }
}
