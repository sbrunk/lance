// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
//
// Apache Lucene reference for Lance's combined_fields (BM25F) query: the JVM
// side of run_combined_fields_compare.sh. Reads the shared two-field corpus and
// query set produced by the `combined_fields_compare` Rust bench, builds an
// in-memory index over `title` + `body` with BM25Similarity(1.2, 0.75), and for
// each query runs a BooleanQuery of one CombinedFieldQuery per term (SHOULD/OR,
// per-field weights from weights.txt). Emits `lucene_topk.txt`: the top-k doc
// ids per query. The driver compares rankings, not absolute scores.
//
// Usage: java -cp <lucene jars>:. LuceneCombinedFieldsBench --in-dir DIR

import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.CombinedFieldQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;

public class LuceneCombinedFieldsBench {

  static String arg(String[] a, String flag, String def) {
    for (int i = 0; i < a.length - 1; i++) {
      if (a[i].equals(flag)) return a[i + 1];
    }
    return def;
  }

  static List<String> readLines(String path) throws Exception {
    List<String> out = new ArrayList<>();
    try (BufferedReader r = new BufferedReader(new FileReader(path))) {
      String line;
      while ((line = r.readLine()) != null) out.add(line);
    }
    return out;
  }

  public static void main(String[] argv) throws Exception {
    String inDir = arg(argv, "--in-dir", "/tmp/combined_fields_compare");
    List<String> titles = readLines(Paths.get(inDir, "title.txt").toString());
    List<String> bodies = readLines(Paths.get(inDir, "body.txt").toString());
    List<String> queries = readLines(Paths.get(inDir, "queries.txt").toString());
    String[] w = readLines(Paths.get(inDir, "weights.txt").toString()).get(0).trim().split("\\s+");
    float wTitle = Float.parseFloat(w[0]);
    float wBody = Float.parseFloat(w[1]);
    int k = Integer.parseInt(w[2]);

    BM25Similarity sim = new BM25Similarity(1.2f, 0.75f);

    // CombinedFieldQuery requires norms; index freqs (no positions needed).
    FieldType textType = new FieldType();
    textType.setTokenized(true);
    textType.setOmitNorms(false);
    textType.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
    textType.freeze();

    Directory dir = new ByteBuffersDirectory();
    IndexWriterConfig iwc = new IndexWriterConfig(new WhitespaceAnalyzer());
    iwc.setSimilarity(sim);
    try (IndexWriter writer = new IndexWriter(dir, iwc)) {
      for (int id = 0; id < titles.size(); id++) {
        Document d = new Document();
        d.add(new StoredField("id", id));
        d.add(new Field("title", titles.get(id), textType));
        d.add(new Field("body", id < bodies.size() ? bodies.get(id) : "", textType));
        writer.addDocument(d);
      }
      writer.commit();
    }

    DirectoryReader reader = DirectoryReader.open(dir);
    IndexSearcher searcher = new IndexSearcher(reader);
    searcher.setSimilarity(sim);
    StoredFields storedFields = searcher.storedFields();

    List<List<Integer>> topk = new ArrayList<>();
    for (String line : queries) {
      BooleanQuery.Builder bq = new BooleanQuery.Builder();
      for (String term : line.trim().split("\\s+")) {
        if (term.isEmpty()) continue;
        CombinedFieldQuery cfq =
            new CombinedFieldQuery.Builder(term)
                .addField("title", wTitle)
                .addField("body", wBody)
                .build();
        bq.add(cfq, BooleanClause.Occur.SHOULD);
      }
      TopDocs td = searcher.search(bq.build(), k);
      List<Integer> ids = new ArrayList<>();
      for (ScoreDoc sd : td.scoreDocs) {
        ids.add(storedFields.document(sd.doc).getField("id").numericValue().intValue());
      }
      topk.add(ids);
    }

    StringBuilder out = new StringBuilder();
    for (List<Integer> ids : topk) {
      for (int j = 0; j < ids.size(); j++) {
        if (j > 0) out.append(' ');
        out.append(ids.get(j));
      }
      out.append('\n');
    }
    Files.writeString(Paths.get(inDir, "lucene_topk.txt"), out.toString());
    System.out.printf("lucene combined_fields: indexed %d docs, ran %d queries (k=%d)%n",
        titles.size(), queries.size(), k);

    reader.close();
    dir.close();
  }
}
