package com.vectordb;

import java.util.ArrayList;
import java.util.List;

/** Port of the C++ chunkText function: splits long text into overlapping word-count chunks. */
public class TextChunker {

    public static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().isEmpty() ? new String[0] : text.trim().split("\\s+");
        if (words.length == 0) return new ArrayList<>();
        if (words.length <= chunkWords) {
            List<String> single = new ArrayList<>();
            single.add(text);
            return single;
        }

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder chunk = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) chunk.append(' ');
                chunk.append(words[j]);
            }
            chunks.add(chunk.toString());
            if (end == words.length) break;
        }
        return chunks;
    }
}
