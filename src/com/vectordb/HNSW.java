package com.vectordb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import com.vectordb.Distances.DistFn;
import com.vectordb.BruteForce.Pair;

/**
 * Port of the C++ HNSW (Hierarchical Navigable Small World) class.
 * std::unordered_map<int, Node> -> HashMap<Integer, Node>.
 * std::mt19937 rng(42) -> java.util.Random(42) (same fixed seed for reproducible demo behavior).
 */
public class HNSW {

    private static class Node {
        VectorItem item;
        int maxLyr;
        List<List<Integer>> nbrs; // nbrs.get(layer) = neighbor ids at that layer

        Node(VectorItem item, int maxLyr, List<List<Integer>> nbrs) {
            this.item = item;
            this.maxLyr = maxLyr;
            this.nbrs = nbrs;
        }
    }

    private final Map<Integer, Node> graph = new HashMap<>();
    private final int M, M0, efBuild;
    private final float mL;
    private int topLayer = -1;
    private int entryPt = -1;
    private final Random rng;

    public HNSW() {
        this(16, 200);
    }

    public HNSW(int m, int efBuild) {
        this.M = m;
        this.M0 = 2 * m;
        this.efBuild = efBuild;
        this.mL = (float) (1.0 / Math.log(m));
        this.rng = new Random(42); // fixed seed, mirrors C++ mt19937 rng(42)
    }

    private int randLevel() {
        double u = rng.nextDouble(); // uniform (0,1)
        if (u <= 0) u = Double.MIN_VALUE; // guard against log(0)
        return (int) Math.floor(-Math.log(u) * mL);
    }

    private static List<List<Integer>> emptyLayers(int lvl) {
        List<List<Integer>> l = new ArrayList<>();
        for (int i = 0; i <= lvl; i++) l.add(new ArrayList<>());
        return l;
    }

    /**
     * Greedy beam search within a single layer.
     * C++ used two priority queues: a min-heap "cands" (frontier to explore) and
     * a max-heap "found" (current best-k, worst on top so it can be evicted).
     */
    private List<Pair> searchLayer(float[] q, int ep, int ef, int lyr, DistFn dist) {
        Set<Integer> visited = new HashSet<>();
        PriorityQueue<Pair> cands = new PriorityQueue<>(); // min-heap: smallest distance first
        PriorityQueue<Pair> found = new PriorityQueue<>(Collections.reverseOrder()); // max-heap: largest first

        float d0 = dist.dist(q, graph.get(ep).item.emb);
        visited.add(ep);
        cands.add(new Pair(d0, ep));
        found.add(new Pair(d0, ep));

        while (!cands.isEmpty()) {
            Pair c = cands.poll();
            if (found.size() >= ef && c.dist > found.peek().dist) break;

            Node cNode = graph.get(c.id);
            if (cNode == null || lyr >= cNode.nbrs.size()) continue;

            for (int nid : cNode.nbrs.get(lyr)) {
                if (visited.contains(nid) || !graph.containsKey(nid)) continue;
                visited.add(nid);
                float nd = dist.dist(q, graph.get(nid).item.emb);
                if (found.size() < ef || nd < found.peek().dist) {
                    cands.add(new Pair(nd, nid));
                    found.add(new Pair(nd, nid));
                    if (found.size() > ef) found.poll();
                }
            }
        }

        List<Pair> res = new ArrayList<>(found);
        res.sort((a, b) -> Float.compare(a.dist, b.dist));
        return res;
    }

    private List<Integer> selectNbrs(List<Pair> cands, int maxM) {
        List<Integer> r = new ArrayList<>();
        int n = Math.min(cands.size(), maxM);
        for (int i = 0; i < n; i++) r.add(cands.get(i).id);
        return r;
    }

    public void insert(VectorItem item, DistFn dist) {
        int id = item.id;
        int lvl = randLevel();
        graph.put(id, new Node(item, lvl, emptyLayers(lvl)));

        if (entryPt == -1) {
            entryPt = id;
            topLayer = lvl;
            return;
        }

        int ep = entryPt;

        // Descend from the top layer down to (lvl+1), keeping only the single closest entry point.
        for (int lc = topLayer; lc > lvl; lc--) {
            Node epNode = graph.get(ep);
            if (lc < epNode.nbrs.size()) {
                List<Pair> w = searchLayer(item.emb, ep, 1, lc, dist);
                if (!w.isEmpty()) ep = w.get(0).id;
            }
        }

        // From min(topLayer, lvl) down to 0: beam-search this layer and connect bidirectionally.
        List<Pair> w = new ArrayList<>();
        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            w = searchLayer(item.emb, ep, efBuild, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            List<Integer> sel = selectNbrs(w, maxM);
            graph.get(id).nbrs.set(lc, sel);

            for (int nid : sel) {
                Node nbrNode = graph.get(nid);
                if (nbrNode == null) continue;
                while (nbrNode.nbrs.size() <= lc) nbrNode.nbrs.add(new ArrayList<>());
                List<Integer> conn = nbrNode.nbrs.get(lc);
                conn.add(id);
                if (conn.size() > maxM) {
                    List<Pair> ds = new ArrayList<>();
                    for (int c : conn) {
                        Node cNode = graph.get(c);
                        if (cNode != null) ds.add(new Pair(dist.dist(nbrNode.item.emb, cNode.item.emb), c));
                    }
                    ds.sort((a, b) -> Float.compare(a.dist, b.dist));
                    conn.clear();
                    for (int i = 0; i < maxM && i < ds.size(); i++) conn.add(ds.get(i).id);
                }
            }

            if (!w.isEmpty()) ep = w.get(0).id;
        }

        if (lvl > topLayer) {
            topLayer = lvl;
            entryPt = id;
        }
    }

    public List<Pair> knn(float[] q, int k, int ef, DistFn dist) {
        if (entryPt == -1) return new ArrayList<>();
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            Node epNode = graph.get(ep);
            if (lc < epNode.nbrs.size()) {
                List<Pair> w = searchLayer(q, ep, 1, lc, dist);
                if (!w.isEmpty()) ep = w.get(0).id;
            }
        }
        List<Pair> w = searchLayer(q, ep, Math.max(ef, k), 0, dist);
        if (w.size() > k) w = new ArrayList<>(w.subList(0, k));
        return w;
    }

    public void remove(int id) {
        if (!graph.containsKey(id)) return;
        for (Node nd : graph.values()) {
            for (List<Integer> layer : nd.nbrs) {
                layer.removeIf(n -> n == id);
            }
        }
        if (entryPt == id) {
            entryPt = -1;
            for (int otherId : graph.keySet()) {
                if (otherId != id) {
                    entryPt = otherId;
                    break;
                }
            }
        }
        graph.remove(id);
    }

    public int size() {
        return graph.size();
    }

    // ── Graph introspection (for /hnsw-info) ─────────────────────────────

    public static class NodeView {
        public int id, maxLyr;
        public String metadata, category;

        public NodeView(int id, String metadata, String category, int maxLyr) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.maxLyr = maxLyr;
        }
    }

    public static class EdgeView {
        public int src, dst, lyr;

        public EdgeView(int src, int dst, int lyr) {
            this.src = src;
            this.dst = dst;
            this.lyr = lyr;
        }
    }

    public static class GraphInfo {
        public int topLayer, nodeCount;
        public List<Integer> nodesPerLayer = new ArrayList<>();
        public List<Integer> edgesPerLayer = new ArrayList<>();
        public List<NodeView> nodes = new ArrayList<>();
        public List<EdgeView> edges = new ArrayList<>();
    }

    public GraphInfo getInfo() {
        GraphInfo gi = new GraphInfo();
        gi.topLayer = topLayer;
        gi.nodeCount = graph.size();
        int maxL = Math.max(topLayer + 1, 1);
        for (int i = 0; i < maxL; i++) {
            gi.nodesPerLayer.add(0);
            gi.edgesPerLayer.add(0);
        }

        for (Map.Entry<Integer, Node> e : graph.entrySet()) {
            int id = e.getKey();
            Node nd = e.getValue();
            gi.nodes.add(new NodeView(id, nd.item.metadata, nd.item.category, nd.maxLyr));
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer.set(lc, gi.nodesPerLayer.get(lc) + 1);
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (id < nid) {
                            gi.edgesPerLayer.set(lc, gi.edgesPerLayer.get(lc) + 1);
                            gi.edges.add(new EdgeView(id, nid, lc));
                        }
                    }
                }
            }
        }
        return gi;
    }
}
