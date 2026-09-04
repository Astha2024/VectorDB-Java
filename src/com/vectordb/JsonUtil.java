package com.vectordb;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Port of the C++ JSON HELPERS section. This is the same minimal, hand-rolled
 * string-scanning approach as the original (no external JSON library), kept
 * deliberately dependency-free to match the spirit of the single-header C++ version.
 */
public class JsonUtil {

    /** Quote and escape a string as a JSON string literal, e.g. jS("hi") -> "\"hi\"". */
    public static String jS(String s) {
        StringBuilder o = new StringBuilder("\"");
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
        return o.append('"').toString();
    }

    /** Render a float[] as a JSON array with 4 decimal places, e.g. [0.1234,0.5678]. */
    public static String jVec(float[] v) {
        StringBuilder ss = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) ss.append(',');
            ss.append(String.format(Locale.ROOT, "%.4f", v[i]));
        }
        return ss.append(']').toString();
    }

    /** Parse a comma-separated list of floats, e.g. "0.1,0.2,0.3" -> [0.1f, 0.2f, 0.3f]. */
    public static float[] parseVec(String s) {
        if (s == null || s.isEmpty()) return new float[0];
        String[] parts = s.split(",");
        List<Float> vals = new ArrayList<>();
        for (String t : parts) {
            try {
                vals.add(Float.parseFloat(t.trim()));
            } catch (NumberFormatException ignored) {
                // matches the C++ try/catch swallow of bad tokens
            }
        }
        float[] out = new float[vals.size()];
        for (int i = 0; i < out.length; i++) out[i] = vals.get(i);
        return out;
    }

    /** Extract a JSON string field's value, handling basic escape sequences. Returns "" if not found. */
    public static String extractStr(String body, String key) {
        if (body == null) return "";
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) return "";
        p = body.indexOf(':', p);
        if (p < 0) return "";
        p++;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                char esc = body.charAt(p);
                switch (esc) {
                    case '"': result.append('"'); break;
                    case '\\': result.append('\\'); break;
                    case 'n': result.append('\n'); break;
                    case 'r': result.append('\r'); break;
                    case 't': result.append('\t'); break;
                    default: result.append(esc);
                }
            } else {
                result.append(c);
            }
            p++;
        }
        return result.toString();
    }

    /** Extract a JSON integer field's value, or def if not found/parseable. */
    public static int extractInt(String body, String key, int def) {
        if (body == null) return def;
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) return def;
        p = body.indexOf(':', p);
        if (p < 0) return def;
        p++;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        int start = p;
        while (p < body.length() && (Character.isDigit(body.charAt(p)) || body.charAt(p) == '-')) p++;
        try {
            return Integer.parseInt(body.substring(start, p));
        } catch (NumberFormatException e) {
            return def;
        }
    }

    /** Extract a JSON array-of-floats field's value, e.g. "embedding":[0.1,0.2] -> [0.1f, 0.2f]. */
    public static float[] extractFloatArray(String body, String key) {
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) return new float[0];
        p = body.indexOf('[', p);
        if (p < 0) return new float[0];
        int e = body.indexOf(']', p);
        if (e < 0) return new float[0];
        return parseVec(body.substring(p + 1, e));
    }

    /** Result of parsing an /insert request body: metadata, category, embedding. */
    public static class InsertBody {
        public String meta = "";
        public String cat = "";
        public float[] emb = new float[0];
        public boolean ok;
    }

    public static InsertBody parseInsertBody(String b) {
        InsertBody r = new InsertBody();
        r.meta = extractStr(b, "metadata");
        r.cat = extractStr(b, "category");
        r.emb = extractFloatArray(b, "embedding");
        r.ok = !r.meta.isEmpty() && r.emb.length > 0;
        return r;
    }
}
