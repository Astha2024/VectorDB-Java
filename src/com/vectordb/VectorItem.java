package com.vectordb;

import java.util.List;

/**
 * Equivalent to the C++ struct VectorItem.
 * Holds one stored vector plus its metadata/category.
 */
public class VectorItem {
    public int id;
    public String metadata;
    public String category;
    public float[] emb;

    public VectorItem(int id, String metadata, String category, float[] emb) {
        this.id = id;
        this.metadata = metadata;
        this.category = category;
        this.emb = emb;
    }

    public VectorItem(int id, String metadata, String category, List<Float> embList) {
        this.id = id;
        this.metadata = metadata;
        this.category = category;
        this.emb = new float[embList.size()];
        for (int i = 0; i < embList.size(); i++) this.emb[i] = embList.get(i);
    }
}
