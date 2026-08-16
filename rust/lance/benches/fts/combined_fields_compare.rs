// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Self-contained `combined_fields` (BM25F) validation harness: Lance vs Lucene.
//!
//! Unlike `mem_wal_fts_bench` (FineWeb + `FtsMemIndex`), this drives the
//! persistent `Dataset` / `InvertedIndex` scanner path a `CombinedFieldsQuery`
//! really takes, over a small synthetic two-field corpus, and cross-checks it
//! against an exact brute-force BM25F oracle and Apache Lucene's
//! `CombinedFieldQuery`.
//!
//! Writes into `--out-dir`:
//!   `title.txt`, `body.txt`   one document per line (whitespace-tokenized)
//!   `queries.txt`             one query per line (space-separated terms)
//!   `weights.txt`             `<w_title> <w_body> <k>`
//!   `lance_topk.txt`          top-k doc ids per query (Lance combined_fields)
//!   `truth.txt`               top-k doc ids per query (exact brute-force BM25F)
//!
//! `LuceneCombinedFieldsBench.java` reads the same corpus and emits
//! `lucene_topk.txt`; `run_combined_fields_compare.sh` reports mutual top-k
//! overlap and recall@k against the brute-force truth.
//!
//! Tokens are lowercase whitespace-separated so Lance's `simple` tokenizer (no
//! stemming / stop words) and Lucene's `WhitespaceAnalyzer` agree, isolating the
//! BM25F scoring. Rankings are compared, not absolute scores: Lance keeps a
//! constant `(k1 + 1)` numerator Lucene lacks, and Lucene quantizes norms.
//!
//! `--perf` also times `combined_fields` against the `best_fields` (MultiMatch)
//! baseline. `combined_fields` runs an exact merged scan with WAND disabled and
//! an O(total docs) length pass, so a large `--docs` (e.g. 200000) is needed to
//! expose its cost relative to the WAND-pruned `best_fields` path.
//!
//! `--skew` swaps the uniform vocab for a Zipfian one and mixes one common head
//! term with one or two rare tail terms per query. That document-frequency skew
//! engages MAXSCORE: the common term's low ceiling falls into the non-essential
//! prefix while the rare terms drive candidate discovery, so `--skew` is where
//! the pruning report below shows the largest factors and `--perf --skew` is
//! where the pruning shows up as latency.
//!
//! Two independent accountings come out of the recall pass:
//!
//! - The scan's own candidate/block/posting counters (see [`pruning_report`]).
//!   These are per-candidate and per-block events, which the production
//!   `MetricsCollector` deliberately does not carry, so the executed plan cannot
//!   report them. The harness gets them by calling the core scan
//!   (`combined_fields_search_with_stats`) directly on the same opened index
//!   segments, behind `lance-index`'s `test-scan-stats` dev-dependency feature.
//! - Whatever the physical plan reports (see [`plan_metrics`]): cache, IO and
//!   timing counters for the real executed plan, which the direct scan call
//!   does not cover.
//!
//! `--flat-baseline` indexes only `title`, so `body` is unindexed and the planner
//! routes the whole corpus to `FlatCombinedFieldsExec`. Comparing a `--perf` run
//! with and without it measures the flat-only path's cost against the indexed
//! plan. `--skip-truth` drops the brute-force oracle when only latency matters.
//!
//! Usage: `combined_fields_compare --out-dir DIR [--docs N] [--vocab V] [--queries Q] [--k K] [--skew] [--flat-baseline] [--skip-truth] [--perf [--perf-iters I]]`
//!
//! Requires the `test-scan-stats` feature on the `lance-index` dev-dependency
//! for the pruning report, and `test-oracle` for the brute-force truth. Both are
//! enabled for this target in `rust/lance/Cargo.toml`.

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{
    ArrayRef, Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use datafusion::physical_plan::metrics::MetricValue;
use datafusion::physical_plan::{ExecutionPlan, displayable};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt, scalar::load_segments};
use lance_core::Result;
use lance_datafusion::exec::{LanceExecutionOptions, execute_plan};
use lance_index::IndexType;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::prefilter::NoFilter;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::document_tokenizer::DocType;
use lance_index::scalar::inverted::oracle::{brute_force_bm25f, brute_force_top_k};
use lance_index::scalar::inverted::query::{
    CombinedFieldsQuery, FtsQuery, FtsSearchParams, MultiMatchQuery, Operator, Tokens,
};
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::inverted::{
    CombinedCorpusStats, CombinedFieldColumn, CombinedScanStats, DocumentGranularity,
    InvertedIndex, build_combined_bm25_scorer, combined_fields_search_with_stats,
};
use lance_tokenizer::Language;

const W_TITLE: f32 = 2.0;
const W_BODY: f32 = 1.0;
/// Deterministic splitmix64-style generator so the corpus and query set are
/// reproducible across the Lance, Lucene, and brute-force sides.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) ^ self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Unnormalized Zipf cumulative weights over `vocab` ranks (rank `r` has weight
/// `1/(r + 1)`), so [`sample_zipf`] can draw ranks with a `1/rank` frequency law.
fn zipf_cumulative(vocab: usize) -> Vec<f64> {
    let mut cumulative = Vec::with_capacity(vocab);
    let mut acc = 0.0f64;
    for rank in 0..vocab {
        acc += 1.0 / (rank as f64 + 1.0);
        cumulative.push(acc);
    }
    cumulative
}

/// Draw a token rank from the Zipf table: rank 0 (the most common token) carries
/// ~`1/H_vocab` of the mass, so a handful of head tokens dominate while the tail
/// stays rare. Deterministic given the `Lcg`.
fn sample_zipf(rng: &mut Lcg, cumulative: &[f64]) -> usize {
    let total = *cumulative.last().expect("non-empty vocab");
    // 53-bit uniform in [0, 1) from the generator, scaled to [0, total).
    let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64 * total;
    match cumulative.binary_search_by(|w| w.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) | Err(i) => i.min(cumulative.len() - 1),
    }
}

/// A synthetic two-field corpus plus the query set, all whitespace-tokenized.
struct Corpus {
    titles: Vec<Vec<String>>,
    bodies: Vec<Vec<String>>,
    queries: Vec<Vec<String>>,
}

fn generate_corpus(docs: usize, vocab: usize, num_queries: usize, skew: bool) -> Corpus {
    if skew {
        return generate_corpus_skewed(docs, vocab, num_queries);
    }
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let token = |i: usize| format!("t{i}");

    let sample = |rng: &mut Lcg, count: usize, skew: usize| -> Vec<String> {
        // `skew` biases token selection so the same term has different document
        // frequencies in `title` vs `body`, the skew BM25F blends.
        (0..count)
            .map(|_| token((rng.below(vocab) + skew * rng.below(3)) % vocab))
            .collect()
    };

    let mut titles = Vec::with_capacity(docs);
    let mut bodies = Vec::with_capacity(docs);
    for _ in 0..docs {
        let title_len = 1 + rng.below(3);
        let body_len = 2 + rng.below(5);
        titles.push(sample(&mut rng, title_len, 0));
        bodies.push(sample(&mut rng, body_len, 1));
    }

    let queries = (0..num_queries)
        .map(|_| {
            let terms = 2 + rng.below(2);
            (0..terms).map(|_| token(rng.below(vocab))).collect()
        })
        .collect();

    Corpus {
        titles,
        bodies,
        queries,
    }
}

/// Skewed variant of [`generate_corpus`]: tokens are drawn Zipfian, so a few head
/// tokens land in a large fraction of docs while a long tail stays rare. Each
/// query mixes one common head term (rank `< HEAD`) with one or two rare tail
/// terms (rank `>= HEAD`), the df skew that engages MAXSCORE. The brute-force
/// oracle recomputes df from this corpus, so it stays exact regardless of the
/// distribution.
fn generate_corpus_skewed(docs: usize, vocab: usize, num_queries: usize) -> Corpus {
    // Common-token ranks are [0, HEAD); every query pulls its one common term
    // from there and its rare terms from the [HEAD, vocab) tail.
    const HEAD: usize = 8;
    let head = HEAD.min(vocab.max(2) / 2).max(1);
    let tail = (vocab - head).max(1);

    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let token = |i: usize| format!("t{i}");
    let cumulative = zipf_cumulative(vocab);
    let sample = |rng: &mut Lcg, count: usize| -> Vec<String> {
        (0..count)
            .map(|_| token(sample_zipf(rng, &cumulative)))
            .collect()
    };

    let mut titles = Vec::with_capacity(docs);
    let mut bodies = Vec::with_capacity(docs);
    for _ in 0..docs {
        let title_len = 1 + rng.below(3);
        let body_len = 2 + rng.below(5);
        titles.push(sample(&mut rng, title_len));
        bodies.push(sample(&mut rng, body_len));
    }

    let queries = (0..num_queries)
        .map(|_| {
            let mut terms = Vec::with_capacity(3);
            terms.push(token(rng.below(head))); // one common head term
            for _ in 0..(1 + rng.below(2)) {
                // one or two rare tail terms
                terms.push(token(head + rng.below(tail)));
            }
            terms
        })
        .collect();

    Corpus {
        titles,
        bodies,
        queries,
    }
}

/// Exact top-k doc ids per query, from the shared brute-force oracle.
fn brute_force_truth(corpus: &Corpus, k: usize) -> Vec<Vec<i32>> {
    let titles: Vec<String> = corpus.titles.iter().map(|doc| doc.join(" ")).collect();
    let bodies: Vec<String> = corpus.bodies.iter().map(|doc| doc.join(" ")).collect();
    let columns = [
        (W_TITLE, titles.iter().map(String::as_str).collect()),
        (W_BODY, bodies.iter().map(String::as_str).collect()),
    ];
    corpus
        .queries
        .iter()
        .map(|query| brute_force_top_k(&brute_force_bm25f(&columns, &query.join(" "), false), k))
        .collect()
}

/// Per-query count of docs matching at least one query term in either field, the
/// candidate set an un-pruned merged scan would score. MAXSCORE's `discovered`
/// counter is compared against this to show discovery pruning.
fn union_sizes(corpus: &Corpus) -> Vec<usize> {
    corpus
        .queries
        .iter()
        .map(|query| {
            let terms: HashSet<&str> = query.iter().map(String::as_str).collect();
            (0..corpus.titles.len())
                .filter(|&doc| {
                    corpus.titles[doc]
                        .iter()
                        .chain(&corpus.bodies[doc])
                        .any(|t| terms.contains(t.as_str()))
                })
                .count()
        })
        .collect()
}

/// Sort `(id, score)` pairs by descending score (id as a stable tiebreak) and
/// return the first `k` ids.
fn top_k_ids(scored: &mut [(i32, f32)], k: usize) -> Vec<i32> {
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.iter().take(k).map(|(id, _)| *id).collect()
}

fn join(tokens: &[Vec<String>]) -> String {
    tokens
        .iter()
        .map(|d| d.join(" "))
        .collect::<Vec<_>>()
        .join("\n")
}

fn write_topk(path: &Path, rows: &[Vec<i32>]) -> Result<()> {
    let mut out = String::new();
    for ids in rows {
        let line = ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        out.push_str(&line);
        out.push('\n');
    }
    std::fs::write(path, out).map_err(|e| lance_core::Error::io(format!("write {path:?}: {e}")))
}

/// The FTS index configuration shared by every column (scoping harness).
fn fts_params() -> InvertedIndexParams {
    InvertedIndexParams::new("simple".to_string(), Language::English)
        .lower_case(true)
        .stem(false)
        .remove_stop_words(false)
        .ascii_folding(false)
        .max_token_length(None)
}

/// (Re)build the FTS indexes on `title` and `body` in place (replace=true).
///
/// With `flat_baseline` only `title` is indexed. `body` then has no inverted
/// index at all, so every target fragment is uncovered for it and the planner
/// routes the whole corpus to `FlatCombinedFieldsExec`. That is the same
/// flat-only whole-corpus route an overlay-stale index forces, without the
/// overlay plumbing.
async fn create_fts_indexes(dataset: &mut Dataset, flat_baseline: bool) -> Result<()> {
    let params = fts_params();
    dataset
        .create_index(&["title"], IndexType::Inverted, None, &params, true)
        .await?;
    if !flat_baseline {
        dataset
            .create_index(&["body"], IndexType::Inverted, None, &params, true)
            .await?;
    }
    Ok(())
}

/// Build a `Dataset` with FTS indexes on `title` and `body`.
async fn build_indexed_dataset(
    dir: &Path,
    corpus: &Corpus,
    write_params: WriteParams,
    flat_baseline: bool,
) -> Result<Dataset> {
    let ids = Int32Array::from((0..corpus.titles.len() as i32).collect::<Vec<_>>());
    let titles = StringArray::from(
        corpus
            .titles
            .iter()
            .map(|d| d.join(" "))
            .collect::<Vec<_>>(),
    );
    let bodies = StringArray::from(
        corpus
            .bodies
            .iter()
            .map(|d| d.join(" "))
            .collect::<Vec<_>>(),
    );
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("body", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ids) as ArrayRef,
            Arc::new(titles) as ArrayRef,
            Arc::new(bodies) as ArrayRef,
        ],
    )?;

    let ds_uri = dir.join("lance_ds");
    let ds_uri = ds_uri.to_str().unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let t = Instant::now();
    let mut dataset = Dataset::write(reader, ds_uri, Some(write_params)).await?;
    let write_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    create_fts_indexes(&mut dataset, flat_baseline).await?;
    let index_ms = t.elapsed().as_secs_f64() * 1e3;
    writeln!(
        std::io::stdout(),
        "build: dataset write {write_ms:.1} ms, fts index {index_ms:.1} ms (flat_baseline={flat_baseline})"
    )
    .map_err(|e| lance_core::Error::io(format!("stdout: {e}")))?;
    Ok(dataset)
}

/// `combined_fields` (BM25F) query over `title`+`body` with the fixed weights.
fn combined_query(query: &[String], k: usize) -> Result<FullTextSearchQuery> {
    let combined = CombinedFieldsQuery::try_new(
        query.join(" "),
        vec!["title".to_string(), "body".to_string()],
    )?
    .try_with_boosts(vec![W_TITLE, W_BODY])?;
    Ok(FullTextSearchQuery::new_query(FtsQuery::CombinedFields(combined)).limit(Some(k as i64)))
}

/// `best_fields` (MultiMatch: per-column BM25, max-fused) with the same weights.
/// Serves as the perf baseline: each column runs a WAND-pruned `MatchQueryExec`.
fn best_fields_query(query: &[String], k: usize) -> Result<FullTextSearchQuery> {
    let multi = MultiMatchQuery::try_new(
        query.join(" "),
        vec!["title".to_string(), "body".to_string()],
    )?
    .try_with_boosts(vec![W_TITLE, W_BODY])?;
    Ok(FullTextSearchQuery::new_query(FtsQuery::MultiMatch(multi)).limit(Some(k as i64)))
}

/// `(id, score)` per hit, plus the executed plan, whose metrics are read back by
/// [`plan_metrics`].
async fn run_fts(
    dataset: &Dataset,
    fts: FullTextSearchQuery,
) -> Result<(Vec<(i32, f32)>, Arc<dyn ExecutionPlan>)> {
    let mut scan = dataset.scan();
    scan.project(&["id"])?.full_text_search(fts)?;
    let plan = scan.create_plan().await?;
    let options = LanceExecutionOptions {
        skip_logging: true,
        ..Default::default()
    };
    let mut stream = execute_plan(plan.clone(), options)?;
    let mut hits = Vec::new();
    while let Some(batch) = stream.try_next().await? {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let scores = batch
            .column_by_name("_score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        hits.extend((0..batch.num_rows()).map(|row| (ids.value(row), scores.value(row))));
    }
    Ok((hits, plan))
}

/// Open the committed inverted-index segments of `title` and `body` as the
/// `&[CombinedFieldColumn]` the core `combined_fields` scan takes, with the same
/// weights [`combined_query`] puts on the query.
///
/// `Ok(None)` when a target column has no FTS index at all: that is the
/// `--flat-baseline` layout, where the planner routes the whole corpus to the
/// flat path and an indexed scan would not describe what ran.
///
/// Mirrors `open_combined_fields_scan`'s indexed arm (row granularity, every
/// committed segment) using only public API, so the counters below come from the
/// same scan the plan runs, over the same inputs.
async fn open_combined_columns(dataset: &Dataset) -> Result<Option<Vec<CombinedFieldColumn>>> {
    let mut columns = Vec::with_capacity(2);
    for (name, weight) in [("title", W_TITLE), ("body", W_BODY)] {
        let Some(segments) = load_segments(dataset, name, DocumentGranularity::Row).await? else {
            return Ok(None);
        };
        let mut indices = Vec::with_capacity(segments.len());
        for segment in &segments {
            let index = dataset
                .open_scalar_index(name, &segment.uuid, &NoOpMetricsCollector)
                .await?;
            let inverted = index
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .ok_or_else(|| {
                    lance_core::Error::invalid_input(format!(
                        "index for column {name} and segment {} is not an inverted index",
                        segment.uuid
                    ))
                })?;
            indices.push(Arc::new(inverted.clone()));
        }
        columns.push(CombinedFieldColumn {
            column: name.to_string(),
            weight,
            indices,
        });
    }
    Ok(Some(columns))
}

/// Sum the scan's own pruning counters over every query, by calling the core
/// scan directly.
///
/// The corpus terms are already what the `simple` tokenizer would produce
/// (lowercase, whitespace-separated, no stemming), so passing the query terms
/// straight through as [`Tokens`] gives the scan the same term set the plan's
/// tokenizer would.
async fn accumulate_scan_stats(
    columns: &[CombinedFieldColumn],
    queries: &[Vec<String>],
    k: usize,
) -> Result<CombinedScanStats> {
    let params = FtsSearchParams::new().with_limit(Some(k));
    let mut total = CombinedScanStats::default();
    for query in queries {
        let tokens = Tokens::new(query.clone(), DocType::Text);
        let scorer =
            build_combined_bm25_scorer(columns, &tokens, CombinedCorpusStats::IndexOnly, None)
                .await?;
        let (_row_ids, _scores, stats) = combined_fields_search_with_stats(
            columns,
            &tokens,
            &params,
            Operator::Or,
            &scorer,
            Arc::new(NoFilter),
            &NoOpMetricsCollector,
        )
        .await?;
        total.discovered += stats.discovered;
        total.pruned += stats.pruned;
        total.scored += stats.scored;
        total.blocks_total += stats.blocks_total;
        total.blocks_read += stats.blocks_read;
        total.postings_total += stats.postings_total;
        total.postings_read += stats.postings_read;
    }
    Ok(total)
}

/// Report how much work the two prunes saved over the whole query set.
///
/// `union` is what an un-pruned merged scan would score, so `union/discovered`
/// is the discovery pruning factor and `scored/discovered` the per-candidate
/// one. `*_read` under `*_total` is what says the block-skipping fast path ran;
/// equality says nothing, since it also decodes everything when nothing is
/// prunable (a single-term query, or one whose terms all stay essential).
fn pruning_report(
    stats: &CombinedScanStats,
    union: usize,
    num_queries: usize,
    docs: usize,
    skew: bool,
) -> String {
    let CombinedScanStats {
        discovered,
        pruned,
        scored,
        blocks_total,
        blocks_read,
        postings_total,
        postings_read,
    } = *stats;
    let nq = num_queries as f64;
    let mode = if skew { "skewed" } else { "uniform" };
    format!(
        "\n=== combined_fields scan work ({mode}, {num_queries} queries, {docs} docs) ===\n\
         candidates: discovered={discovered} pruned={pruned} scored={scored} union={union}\n\
         avg/q:      discovered={:.1} scored={:.1} union={:.1}\n\
         discovery pruning: union/discovered = {:.2}x fewer candidates walked\n\
         blocks:     read={blocks_read} of {blocks_total} ({:.3} kept)\n\
         postings:   read={postings_read} of {postings_total} ({:.3} kept)",
        discovered as f64 / nq,
        scored as f64 / nq,
        union as f64 / nq,
        union as f64 / discovered.max(1) as f64,
        blocks_read as f64 / blocks_total.max(1) as f64,
        postings_read as f64 / postings_total.max(1) as f64,
    )
}

/// One metric name's total over a plan tree.
#[derive(Default, Clone, Copy)]
struct ScanMetric {
    /// A count for counters and gauges, nanoseconds for time-valued metrics.
    total: usize,
    is_time: bool,
}

/// Sum every metric the plan tree reports, keyed by name. DataFusion reports one
/// metric set per operator per partition, so same-named values are added up.
///
/// Deliberately name-agnostic: the harness prints whatever the FTS operators
/// expose rather than a fixed list, so it keeps working when that surface changes.
/// Today it is the baseline metrics (`output_rows`, `elapsed_compute`) plus the
/// coarse index counters (`parts_loaded`, `index_cache_hits`/`_misses`,
/// `partitions_searched`) and the scorer/segment timings.
///
/// Complements [`pruning_report`] rather than overlapping it: this side covers
/// the plan that really executed, including the flat route under
/// `--flat-baseline`, and reports cache/IO/timing, none of which the scan's
/// pruning counters carry.
fn plan_metrics(plan: &Arc<dyn ExecutionPlan>) -> BTreeMap<String, ScanMetric> {
    let mut totals = BTreeMap::new();
    collect_plan_metrics(plan, &mut totals);
    totals
}

fn collect_plan_metrics(plan: &Arc<dyn ExecutionPlan>, totals: &mut BTreeMap<String, ScanMetric>) {
    if let Some(metrics) = plan.metrics() {
        for metric in metrics.iter() {
            let value = metric.value();
            // Wall-clock start/end instants: nothing to sum across queries.
            if matches!(
                value,
                MetricValue::StartTimestamp(_) | MetricValue::EndTimestamp(_)
            ) {
                continue;
            }
            let entry = totals.entry(value.name().to_string()).or_default();
            entry.total += value.as_usize();
            entry.is_time = matches!(
                value,
                MetricValue::ElapsedCompute(_) | MetricValue::Time { .. }
            );
        }
    }
    for child in plan.children() {
        collect_plan_metrics(child, totals);
    }
}

/// Accumulate one query's plan metrics into a running total.
fn merge_plan_metrics(totals: &mut BTreeMap<String, ScanMetric>, plan: &Arc<dyn ExecutionPlan>) {
    for (name, metric) in plan_metrics(plan) {
        let entry = totals.entry(name).or_default();
        entry.total += metric.total;
        entry.is_time = metric.is_time;
    }
}

/// Print the physical plan of the `combined_fields` query for one query, so the
/// flat-vs-indexed route can be confirmed before trusting any timing.
async fn print_combined_plan(dataset: &Dataset, query: &[String], k: usize) -> Result<()> {
    let (_, plan) = run_fts(dataset, combined_query(query, k)?).await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    writeln!(
        std::io::stdout(),
        "\n=== combined_fields physical plan (query: {}) ===\n{rendered}",
        query.join(" ")
    )
    .map_err(|e| lance_core::Error::io(format!("stdout: {e}")))?;
    Ok(())
}

/// Print the plan actually taken, then time it. A no-op without `--perf`.
async fn run_perf(
    dataset: &Dataset,
    corpus: &Corpus,
    k: usize,
    perf: bool,
    perf_iters: usize,
) -> Result<()> {
    if !perf {
        return Ok(());
    }
    if let Some(query) = corpus.queries.first() {
        print_combined_plan(dataset, query, k).await?;
    }
    bench_latency(dataset, corpus, k, perf_iters).await
}

/// Mean, p50, and p95 of a latency sample (ms). Sorts `samples` in place.
fn latency_stats(samples: &mut [f64]) -> (f64, f64, f64) {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = |q: f64| samples[((samples.len() as f64 * q) as usize).min(samples.len() - 1)];
    (mean, at(0.50), at(0.95))
}

/// Time `combined_fields` against the `best_fields` baseline on the same indexed
/// dataset, interleaving the two per query so system noise hits both equally,
/// and print mean/p50/p95 latency + the combined/best slowdown ratio.
async fn bench_latency(dataset: &Dataset, corpus: &Corpus, k: usize, iters: usize) -> Result<()> {
    // Warm caches (index metadata, doc lengths) for both paths before timing.
    let mut sink = 0usize;
    for query in &corpus.queries {
        sink += run_fts(dataset, combined_query(query, k)?).await?.0.len();
        sink += run_fts(dataset, best_fields_query(query, k)?)
            .await?
            .0
            .len();
    }

    let mut combined_ms = Vec::with_capacity(iters * corpus.queries.len());
    let mut best_ms = Vec::with_capacity(iters * corpus.queries.len());
    for _ in 0..iters {
        for query in &corpus.queries {
            let bfq = best_fields_query(query, k)?;
            let t = Instant::now();
            sink += run_fts(dataset, bfq).await?.0.len();
            best_ms.push(t.elapsed().as_secs_f64() * 1e3);

            let cfq = combined_query(query, k)?;
            let t = Instant::now();
            sink += run_fts(dataset, cfq).await?.0.len();
            combined_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    std::hint::black_box(sink);

    let (b_mean, b_p50, b_p95) = latency_stats(&mut best_ms);
    let (c_mean, c_p50, c_p95) = latency_stats(&mut combined_ms);
    let b_qps = 1e3 / b_mean;
    let c_qps = 1e3 / c_mean;
    let slow_mean = c_mean / b_mean;
    let slow_p50 = c_p50 / b_p50;
    let ndocs = corpus.titles.len();
    let nq = corpus.queries.len();
    let report = format!(
        "\n=== FTS multi-column latency: best_fields vs combined_fields ===\n\
         corpus {ndocs} docs, {nq} queries, k={k}, {iters} iters\n\
         {:<16} {:>10} {:>10} {:>10} {:>12}\n\
         {:<16} {b_mean:>10.3} {b_p50:>10.3} {b_p95:>10.3} {b_qps:>12.1}\n\
         {:<16} {c_mean:>10.3} {c_p50:>10.3} {c_p95:>10.3} {c_qps:>12.1}\n\
         combined/best slowdown: {slow_mean:.2}x (mean), {slow_p50:.2}x (p50)",
        "mode", "mean(ms)", "p50(ms)", "p95(ms)", "QPS", "best_fields", "combined_fields",
    );
    writeln!(std::io::stdout(), "{report}")
        .map_err(|e| lance_core::Error::io(format!("stdout: {e}")))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run(
    out_dir: &Path,
    docs: usize,
    vocab: usize,
    num_queries: usize,
    k: usize,
    perf: bool,
    perf_iters: usize,
    skew: bool,
    stable_row_ids: bool,
    max_rows_per_file: Option<usize>,
    compact: bool,
    flat_baseline: bool,
    skip_truth: bool,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)
        .map_err(|e| lance_core::Error::io(format!("mkdir {out_dir:?}: {e}")))?;
    let t = Instant::now();
    let corpus = generate_corpus(docs, vocab, num_queries, skew);
    writeln!(
        std::io::stdout(),
        "build: corpus generate {:.1} ms ({docs} docs)",
        t.elapsed().as_secs_f64() * 1e3
    )
    .map_err(|e| lance_core::Error::io(format!("stdout: {e}")))?;

    std::fs::write(out_dir.join("title.txt"), join(&corpus.titles))
        .map_err(|e| lance_core::Error::io(format!("write title.txt: {e}")))?;
    std::fs::write(out_dir.join("body.txt"), join(&corpus.bodies))
        .map_err(|e| lance_core::Error::io(format!("write body.txt: {e}")))?;
    std::fs::write(
        out_dir.join("queries.txt"),
        corpus
            .queries
            .iter()
            .map(|q| q.join(" "))
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .map_err(|e| lance_core::Error::io(format!("write queries.txt: {e}")))?;
    std::fs::write(
        out_dir.join("weights.txt"),
        format!("{W_TITLE} {W_BODY} {k}\n"),
    )
    .map_err(|e| lance_core::Error::io(format!("write weights.txt: {e}")))?;

    let write_params = WriteParams {
        enable_stable_row_ids: stable_row_ids,
        max_rows_per_file: max_rows_per_file.unwrap_or(WriteParams::default().max_rows_per_file),
        ..Default::default()
    };
    let mut dataset = build_indexed_dataset(out_dir, &corpus, write_params, flat_baseline).await?;
    if compact {
        compact_files(&mut dataset, CompactionOptions::default(), None).await?;
        // Rebuild the FTS indexes so the builder re-scans the compacted physical
        // layout and re-captures row ids in the new scan order.
        create_fts_indexes(&mut dataset, flat_baseline).await?;
    }
    if skip_truth {
        return run_perf(&dataset, &corpus, k, perf, perf_iters).await;
    }
    let truth = brute_force_truth(&corpus, k);
    // Plan metrics for exactly this recall pass, one `combined_fields_search` per query.
    let mut scan_metrics = BTreeMap::new();
    let mut lance = Vec::with_capacity(corpus.queries.len());
    for query in &corpus.queries {
        let (mut scored, plan) = run_fts(&dataset, combined_query(query, k)?).await?;
        merge_plan_metrics(&mut scan_metrics, &plan);
        lance.push(top_k_ids(&mut scored, k));
    }
    write_topk(&out_dir.join("truth.txt"), &truth)?;
    write_topk(&out_dir.join("lance_topk.txt"), &lance)?;

    // Recall of Lance vs the exact truth, reported inline for a quick signal.
    let mut hit = 0usize;
    let mut total = 0usize;
    for (l, t) in lance.iter().zip(&truth) {
        let tset: HashSet<i32> = t.iter().copied().collect();
        hit += l.iter().filter(|id| tset.contains(id)).count();
        total += t.len();
    }
    let recall = if total > 0 {
        hit as f64 / total as f64
    } else {
        0.0
    };
    let mut stdout = std::io::stdout();
    writeln!(
        stdout,
        "lance combined_fields recall@{k} vs brute-force BM25F = {recall:.4} ({docs} docs, {num_queries} queries)"
    )
    .map_err(|e| lance_core::Error::io(format!("stdout: {e}")))?;

    // The scan's own pruning counters, from the same query set the recall pass
    // above ran. Skipped when a target column has no index: the plan then takes
    // the flat route, which has no candidate discovery to prune.
    let nq = corpus.queries.len() as f64;
    match open_combined_columns(&dataset).await? {
        Some(columns) => {
            let stats = accumulate_scan_stats(&columns, &corpus.queries, k).await?;
            let union: usize = union_sizes(&corpus).iter().sum();
            writeln!(
                stdout,
                "{}",
                pruning_report(&stats, union, num_queries, docs, skew)
            )
            .map_err(|e| lance_core::Error::io(format!("stdout: {e}")))?;
        }
        None => writeln!(
            stdout,
            "\nscan work: not reported, a target column has no FTS index (flat route)"
        )
        .map_err(|e| lance_core::Error::io(format!("stdout: {e}")))?,
    }

    // Whatever the FTS operators report for the recall pass above, which is the
    // cache/IO/timing side rather than the pruning counters printed above.
    writeln!(
        stdout,
        "\n=== combined_fields plan metrics ({num_queries} queries) ==="
    )
    .map_err(|e| lance_core::Error::io(format!("stdout: {e}")))?;
    for (name, metric) in &scan_metrics {
        let line = if metric.is_time {
            let ms = metric.total as f64 / 1e6;
            format!("{name:<28} {ms:>12.3} ms total {:>10.3} ms/q", ms / nq)
        } else {
            format!(
                "{name:<28} {:>12} total {:>13.1} /q",
                metric.total,
                metric.total as f64 / nq
            )
        };
        writeln!(stdout, "{line}").map_err(|e| lance_core::Error::io(format!("stdout: {e}")))?;
    }

    run_perf(&dataset, &corpus, k, perf, perf_iters).await
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args()
        .skip(1)
        .filter(|s| s != "--bench")
        .collect();
    let get = |flag: &str, def: &str| -> String {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1))
            .cloned()
            .unwrap_or_else(|| def.to_string())
    };
    let out_dir = get("--out-dir", "/tmp/combined_fields_compare");
    let docs: usize = get("--docs", "1000").parse().unwrap_or(1000);
    let vocab: usize = get("--vocab", "40").parse().unwrap_or(40);
    let num_queries: usize = get("--queries", "40").parse().unwrap_or(40);
    let k: usize = get("--k", "10").parse().unwrap_or(10);
    let perf = argv.iter().any(|a| a == "--perf");
    let perf_iters: usize = get("--perf-iters", "20").parse().unwrap_or(20);
    let skew = argv.iter().any(|a| a == "--skew");
    let stable_row_ids = argv.iter().any(|a| a == "--stable-row-ids");
    let max_rows_per_file: Option<usize> = argv
        .iter()
        .position(|a| a == "--max-rows-per-file")
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse().ok());
    let compact = argv.iter().any(|a| a == "--compact");
    // Index only `title`, leaving `body` unindexed, so the whole corpus takes the
    // flat-only `FlatCombinedFieldsExec` route.
    let flat_baseline = argv.iter().any(|a| a == "--flat-baseline");
    // Skip the brute-force oracle and the union-size pass, both O(docs * queries)
    // outside the timed loop, when only latency is wanted.
    let skip_truth = argv.iter().any(|a| a == "--skip-truth");

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| lance_core::Error::io(format!("runtime: {e}")))?;
    rt.block_on(run(
        Path::new(&out_dir),
        docs,
        vocab,
        num_queries,
        k,
        perf,
        perf_iters,
        skew,
        stable_row_ids,
        max_rows_per_file,
        compact,
        flat_baseline,
        skip_truth,
    ))
}
