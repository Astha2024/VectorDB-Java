package com.vectordb;

public class DocItem {
    public int id;
    public String title;
    public String text;
    public float[] emb;

    public DocItem(int id, String title, String text, float[] emb) {
        this.id = id;
        this.title = title;
        this.text = text;
        this.emb = emb;
    }
}
