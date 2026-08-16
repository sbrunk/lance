// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The indexed `combined_fields` entry point: load every term's postings across
//! the target columns, merge them into the shared row-id space, and rank them
//! with MAXSCORE.

use std::cmp::Reverse;
use std::sync::Arc;

use lance_core::Result;
use lance_core::utils::tokio::spawn_cpu;

use super::super::documents::AddressKeyedDocuments;
use super::super::query::{FtsSearchParams, Operator, Tokens};
use super::super::scorer::CombinedFieldsBM25Scorer;
use super::cursor::{
    LazyTerm, LoadedSource, MaterializedTerm, build_lazy_term, build_materialized_term,
    into_compressed_sources, record_fast_reads,
};
use super::maxscore::{CombinedScanStats, combined_maxscore};
use super::{CombinedFieldColumn, unique_terms};
use crate::metrics::MetricsCollector;
use crate::prefilter::PreFilter;
use crate::vector::graph::OrderedFloat;

/// A scored candidate ordered the way callers read results: `score DESC, row_id
/// ASC`, the same order the single-column path imposes in
/// `classify_wand_exactness_certificate`.
///
/// [`ScoredDoc`](super::super::builder::ScoredDoc) compares on score alone, which
/// is not enough for the bounded heap [`combined_maxscore`] collects into: among
/// equal scores it evicts whichever row the heap happens to hold at the bottom, so
/// rows that belong in the top-k are dropped and no later sort can bring them
/// back. A fully covered plan returns this search's output directly, with no
/// `SortExec` above it, so the order and the membership both have to be settled
/// here.
///
/// `row_id` is stored reversed so that the derived lexicographic ordering ranks a
/// higher row id lower. The heap's smallest element is then the lowest score with
/// the highest row id, which is exactly the candidate a full heap should evict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct RankedDoc {
    pub(super) score: OrderedFloat,
    pub(super) row_id: Reverse<u64>,
}

impl RankedDoc {
    pub(super) fn new(row_id: u64, score: f32) -> Self {
        Self {
            score: OrderedFloat(score),
            row_id: Reverse(row_id),
        }
    }
}

/// Exact cross-field BM25F search over the target columns.
///
/// Loads each query term's postings across every column and partition, then
/// ranks them with the term-at-a-time `combined_maxscore`, which skips scoring
/// a candidate whose upper bound cannot reach the running k-th score. That bound
/// only ever drops a candidate the collector would have rejected anyway, so the
/// top-k is the exhaustive scan's top-k.
///
/// Results come back ordered by `score DESC, row_id ASC`, which is a total order,
/// so the same data always yields the same top-k even when scores tie. Candidates
/// are discovered in ascending row-id order and [`RankedDoc`] settles ties, so
/// neither the membership nor the order depends on heap internals.
///
/// When every partition is fast-path eligible (compressed postings with strictly
/// ascending `row_ids`, so doc-id order equals row-id order and each row owns one
/// doc) the term cursors are block-skipping `LazyTerm`s, and a non-essential
/// term's blocks are decoded only around the sparse candidates the essential
/// terms surface, pruning reads and not just scoring. A legacy partition,
/// unsorted `row_ids`, or a non-compressed posting forces the full-read
/// `MaterializedTerm` fallback, whose result is bit-identical.
///
/// `operator` applies across the virtual field: `And` keeps only docs where
/// every query term appears in at least one column; `Or` keeps docs matching
/// any term. Per-column `boost` is folded into `tf'`.
pub async fn combined_fields_search(
    columns: &[CombinedFieldColumn],
    tokens: &Tokens,
    params: &FtsSearchParams,
    operator: Operator,
    scorer: &CombinedFieldsBM25Scorer,
    prefilter: Arc<dyn PreFilter>,
    metrics: &dyn MetricsCollector,
) -> Result<(Vec<u64>, Vec<f32>)> {
    let (row_ids, scores, _stats) = combined_fields_search_with_stats(
        columns, tokens, params, operator, scorer, prefilter, metrics,
    )
    .await?;
    Ok((row_ids, scores))
}

/// [`combined_fields_search`] plus the [`CombinedScanStats`] of the scan, for a
/// bench that wants to report how much candidate scoring and how much reading
/// the pruning saved.
///
/// This is the whole implementation; [`combined_fields_search`] is the same call
/// with the stats dropped. The stats are counters both arms keep either way, so
/// the default path does no extra work.
///
/// `*_read` below `*_total` is the only thing that says the block-skip fast path
/// pruned reads; equality is ambiguous, since the fast path also decodes
/// everything when nothing is prunable. Tests assert on it, because a gate that
/// never fires leaves every score correct and silently disables the pruning.
///
/// Reachable outside the crate only under `cfg(test)` or the `test-scan-stats`
/// feature: per-candidate, per-block and per-posting counters are not something
/// the production [`MetricsCollector`] carries, which must stay coarse-grained.
pub async fn combined_fields_search_with_stats(
    columns: &[CombinedFieldColumn],
    tokens: &Tokens,
    params: &FtsSearchParams,
    operator: Operator,
    scorer: &CombinedFieldsBM25Scorer,
    prefilter: Arc<dyn PreFilter>,
    metrics: &dyn MetricsCollector,
) -> Result<(Vec<u64>, Vec<f32>, CombinedScanStats)> {
    let terms = unique_terms(tokens);
    let limit = params.limit.unwrap_or(usize::MAX);
    if terms.is_empty() || limit == 0 {
        return Ok((Vec::new(), Vec::new(), CombinedScanStats::default()));
    }

    let mask = prefilter.mask();
    let require_all_terms = operator == Operator::And;

    // Load every term's postings across all columns, in the canonical
    // column → index → partition order, deciding fast-path eligibility as we
    // go. The length sources are collected in that same order so `dl'` sums in
    // the exact scan's order too (float addition is order-sensitive; matching
    // the order keeps every score bit-identical).
    let mut loaded: Vec<Vec<LoadedSource>> = (0..terms.len()).map(|_| Vec::new()).collect();
    let mut length_sources: Vec<(f32, AddressKeyedDocuments)> = Vec::new();
    // Block skipping needs doc-id order to equal row-id order in every partition,
    // which a legacy layout or unsorted `row_ids` breaks. The remaining
    // requirement, that every posting be compressed, is settled below by
    // retyping the loads.
    let mut layout_allows_skipping = true;
    for column in columns {
        let weight = column.weight;
        for index in &column.indices {
            for partition in &index.partitions {
                let docs = partition.docs.address_keyed().await?;
                let is_legacy = partition.is_legacy();
                if is_legacy || !docs.addresses_strictly_ascending() {
                    layout_allows_skipping = false;
                }
                for (term_index, term) in terms.iter().enumerate() {
                    let Some(token_id) = partition.tokens.get(term) else {
                        continue;
                    };
                    let posting = partition
                        .inverted_list
                        .posting_list(token_id, false, metrics)
                        .await?;
                    loaded[term_index].push(LoadedSource {
                        weight,
                        docs: docs.clone(),
                        is_legacy,
                        posting,
                    });
                }
                length_sources.push((weight, docs));
            }
        }
    }
    // `Ok` is the read-pruning fast path, `Err` hands the same loads to the eager
    // fallback. The returned stats narrow this down in one direction only:
    // `blocks_read` under `blocks_total` proves the fast path ran, while equality
    // is ambiguous, since the fast path also decodes everything when nothing is
    // prunable.
    let sources = if layout_allows_skipping {
        into_compressed_sources(loaded)
    } else {
        Err(loaded)
    };

    // Everything past the loads is uninterruptible CPU work: the fallback builds
    // and sorts a per-term `HashMap`, and both paths then run the whole MAXSCORE
    // loop with no await. Offload it so a large query cannot hold a DataFusion
    // worker past a stream drop or task cancellation, matching how the
    // single-column `InvertedIndex::bm25_search` dispatches its per-partition
    // scoring and how `flat_combined_fields_search_stream` dispatches its own
    // scoring loop. The `'static` closure clones the borrowed `scorer` (a handful
    // of per-term statistics) and moves everything else in.
    let scorer = Arc::new(scorer.clone());
    let (top, stats) = spawn_cpu(move || {
        let dl_prime = |row_id: u64| -> f32 {
            length_sources
                .iter()
                .map(|(weight, docs)| weight * docs.doc_length_at(row_id) as f32)
                .sum()
        };
        // Both arms accumulate into one `stats`: the cursors account the reads,
        // `combined_maxscore` the candidates.
        //
        // Cursors that are all exhausted need no special case: `combined_maxscore`
        // finds no candidate and returns an empty heap, which unzips to no hits.
        let mut stats = CombinedScanStats::default();
        let top = match sources {
            Ok(compressed) => {
                let mut cursors: Vec<LazyTerm> = terms
                    .iter()
                    .zip(compressed)
                    .map(|(term, sources)| build_lazy_term(term, sources, &mask, scorer.as_ref()))
                    .collect();
                let top = combined_maxscore(
                    &mut cursors,
                    dl_prime,
                    limit,
                    require_all_terms,
                    scorer.as_ref(),
                    &mut stats,
                );
                // Only the lazy cursors know what they skipped, so their counters
                // are read off after the loop.
                record_fast_reads(&cursors, &mut stats);
                top
            }
            Err(loaded) => {
                let mut cursors: Vec<MaterializedTerm> = Vec::with_capacity(terms.len());
                for (term, sources) in terms.iter().zip(loaded) {
                    cursors.push(build_materialized_term(
                        term,
                        sources,
                        &mask,
                        scorer.as_ref(),
                        &mut stats,
                    ));
                }
                combined_maxscore(
                    &mut cursors,
                    dl_prime,
                    limit,
                    require_all_terms,
                    scorer.as_ref(),
                    &mut stats,
                )
            }
        };
        Result::Ok((top, stats))
    })
    .await?;

    // Ascending in `Reverse<RankedDoc>` is descending in `RankedDoc`, i.e. best
    // first: highest score, and within a score the lowest row id.
    let (row_ids, scores) = top
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse(doc)| (doc.row_id.0, doc.score.0))
        .unzip();
    Ok((row_ids, scores, stats))
}

#[cfg(test)]
mod tests {
    use super::super::super::encoding::MAX_POSTING_BLOCK_SIZE;
    use super::super::super::index::InvertedListFormatVersion;
    use super::super::super::scorer::idf;
    use super::super::super::tokenizer::document_tokenizer::DocType;
    use super::super::stats::{CombinedCorpusStats, build_combined_bm25_scorer};
    use super::super::testing::{
        ElementRows, as_row_documents, combined_columns, combined_top_k, combined_top_k_with_stats,
        element_document_index,
    };
    use super::*;
    use crate::metrics::NoOpMetricsCollector;
    use crate::prefilter::NoFilter;
    use rstest::rstest;

    /// `title` is a `List<String>` where row 0 holds ten `"alpha"` elements, row 1
    /// one `"beta"`, and rows 2..10 one `"gamma"`; `body` holds one `"zzz"` per
    /// row. Row 0 matches `"alpha"` ten times over, so it must win
    /// `"alpha beta"` at `limit = 1`.
    ///
    /// With document-granularity statistics it does not: `docCount'` counts the
    /// 19 title elements instead of the 10 rows and `docFreq'("alpha")` counts 10
    /// element postings instead of 1 row, which collapses `alpha`'s `idf'` and
    /// inflates `beta`'s enough to invert the ranking (row 1 at ~2.2984569). Row
    /// granularity restores row 0 at ~3.1963050, bit-identical to the same data
    /// indexed one document per row.
    #[rstest]
    #[case::v1(InvertedListFormatVersion::V1)]
    #[case::v2(InvertedListFormatVersion::V2)]
    #[tokio::test]
    async fn test_combined_fields_scores_legacy_list_elements_by_row(
        #[case] format_version: InvertedListFormatVersion,
    ) {
        let vocab = ["alpha", "beta", "gamma", "zzz"];
        let mut title: ElementRows = vec![vec![vec!["alpha"]; 10], vec![vec!["beta"]]];
        title.extend((2..10).map(|_| vec![vec!["gamma"]]));
        let body: ElementRows = (0..10).map(|_| vec![vec!["zzz"]]).collect();

        let (title_index, _title_dir) =
            element_document_index(format_version, &vocab, &title).await;
        let (body_index, _body_dir) = element_document_index(format_version, &vocab, &body).await;

        // Precondition: the fixture really is element-per-document, or the test
        // covers nothing.
        let title_docs = title_index.partitions[0]
            .docs
            .address_keyed()
            .await
            .unwrap();
        assert_eq!(title_docs.len(), 19, "one document per title list element");
        assert_eq!(title_docs.num_distinct_rows(), 10);
        assert!(
            !title_docs.addresses_strictly_ascending(),
            "duplicate row addresses must reject the read-pruning fast path, whose `tf'` \
             counts one posting per partition per row",
        );

        // The single-column path reads document granularity; only the cross-field
        // path reads row granularity.
        let terms = ["alpha".to_owned(), "beta".to_owned()];
        assert_eq!(
            title_index
                .bm25_stats_for_terms(&terms, None)
                .await
                .unwrap(),
            (19, 19, vec![10, 1]),
            "document-granularity statistics must keep counting elements",
        );
        assert_eq!(
            title_index
                .bm25_row_stats_for_terms(&terms, None)
                .await
                .unwrap(),
            (19, 10, vec![1, 1]),
            "row-granularity statistics must count distinct row ids",
        );

        let legacy = combined_top_k(
            &combined_columns(vec![title_index, body_index]),
            &["alpha", "beta"],
            1,
        )
        .await;
        assert_eq!(legacy.len(), 1);
        assert_eq!(legacy[0].0, 0, "row 0 matches `alpha` ten times over");
        assert!(
            (legacy[0].1 - 3.196_305).abs() < 1e-5,
            "unexpected score {}",
            legacy[0].1
        );

        // Reindexing the same data one document per row must agree bit for bit.
        let (row_title, _row_title_dir) = element_document_index(
            InvertedListFormatVersion::V3,
            &vocab,
            &as_row_documents(&title),
        )
        .await;
        let (row_body, _row_body_dir) = element_document_index(
            InvertedListFormatVersion::V3,
            &vocab,
            &as_row_documents(&body),
        )
        .await;
        let rebuilt = combined_top_k(
            &combined_columns(vec![row_title, row_body]),
            &["alpha", "beta"],
            1,
        )
        .await;
        assert_eq!(
            legacy
                .iter()
                .map(|(row_id, score)| (*row_id, score.to_bits()))
                .collect::<Vec<_>>(),
            rebuilt
                .iter()
                .map(|(row_id, score)| (*row_id, score.to_bits()))
                .collect::<Vec<_>>(),
            "legacy index must score like the same data reindexed per row",
        );
    }

    /// The single-column path stays at document granularity. It reads
    /// [`InvertedIndex::bm25_stats_for_terms`] via `build_global_bm25_scorer` and
    /// wand scores one posting per document, so a legacy list index is
    /// self-consistent there: `docCount` and `docFreq` count elements, `tf` is one
    /// element's frequency and `dl` one element's length. V1/V2 are released
    /// stable formats, so these `(row_id, score)` bits pin that behavior; moving
    /// it is a separate compatibility decision.
    ///
    /// The element domain also shows through in the results: row 0's ten `"alpha"`
    /// documents surface as ten separate hits at the same row id.
    #[rstest]
    #[case::v1(InvertedListFormatVersion::V1)]
    #[case::v2(InvertedListFormatVersion::V2)]
    #[tokio::test]
    async fn test_single_column_search_keeps_element_granularity(
        #[case] format_version: InvertedListFormatVersion,
    ) {
        let vocab = ["alpha", "beta", "gamma"];
        let mut title: ElementRows = vec![vec![vec!["alpha"]; 10], vec![vec!["beta"]]];
        title.extend((2..10).map(|_| vec![vec!["gamma"]]));
        let (index, _dir) = element_document_index(format_version, &vocab, &title).await;

        let terms = ["alpha".to_owned(), "beta".to_owned()];
        assert_eq!(
            index.bm25_stats_for_terms(&terms, None).await.unwrap(),
            (19, 19, vec![10, 1]),
            "the single-column path's statistics stay at document granularity",
        );

        let tokens = Arc::new(Tokens::new(terms.to_vec(), DocType::Text));
        let params = Arc::new(FtsSearchParams::new().with_limit(Some(4)));
        let scorer = crate::scalar::inverted::build_global_bm25_scorer(
            std::slice::from_ref(&index),
            &tokens,
            &params,
            None,
        )
        .await
        .unwrap();
        let (row_ids, scores) = index
            .bm25_search(
                tokens,
                params,
                Operator::Or,
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
                Some(&scorer),
            )
            .await
            .unwrap();
        // `beta` (row 1) leads on its inflated element-level idf, and row 0
        // repeats once per matching element: the element domain showing through
        // in the hits themselves.
        assert_eq!(row_ids, vec![1, 0, 0, 0]);
        // Every element is one token long and `avgdl` is the element average
        // (19 tokens / 19 documents), so `doc_weight` is exactly 1 and each score
        // is the bare element-granularity `idf`: `ln(1 + (19 - df + 0.5)/(df + 0.5))`
        // with df = 1 for `beta` and df = 10 for `alpha`.
        assert_eq!(
            scores.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            vec![0x4025_c6f0, 0x3f24_f495, 0x3f24_f495, 0x3f24_f495],
        );
        assert_eq!(scores[0].to_bits(), idf(1, 19).to_bits());
        assert_eq!(scores[1].to_bits(), idf(10, 19).to_bits());
    }

    /// Discovery pruning through the real loading path, not just the
    /// `combined_maxscore` unit fixtures: one rare term and one term in every
    /// row, so the common term's ceiling drops below the k-th score at once and
    /// candidate discovery walks the rare term's single posting alone.
    ///
    /// Also the coverage for [`combined_fields_search_with_stats`] and its
    /// re-export: it is reached here through the gated crate path the bench uses,
    /// both counter groups included, and its top-k must be what plain
    /// [`combined_fields_search`] returns.
    #[tokio::test]
    async fn test_combined_fields_search_with_stats_prunes_discovery() {
        use crate::scalar::inverted::{CombinedScanStats, combined_fields_search_with_stats};

        const ROWS: usize = 40;
        let vocab = ["rare", "common"];
        let rows: ElementRows = (0..ROWS)
            .map(|row| {
                if row == 0 {
                    vec![vec!["rare", "common"]]
                } else {
                    vec![vec!["common"]]
                }
            })
            .collect();
        let (index, _dir) =
            element_document_index(InvertedListFormatVersion::V3, &vocab, &rows).await;
        let columns = combined_columns(vec![index]);

        let tokens = Tokens::new(vec!["rare".to_owned(), "common".to_owned()], DocType::Text);
        let scorer =
            build_combined_bm25_scorer(&columns, &tokens, CombinedCorpusStats::IndexOnly, None)
                .await
                .unwrap();
        let params = FtsSearchParams::new().with_limit(Some(1));

        let (row_ids, scores, stats): (Vec<u64>, Vec<f32>, CombinedScanStats) =
            combined_fields_search_with_stats(
                &columns,
                &tokens,
                &params,
                Operator::Or,
                &scorer,
                Arc::new(NoFilter),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        assert_eq!(row_ids, vec![0], "the only row holding `rare` must win");
        // Every candidate discovered is either pruned or scored, so the counts
        // account for the whole loop.
        assert_eq!(stats.discovered, stats.pruned + stats.scored);
        assert_eq!(
            stats.discovered, 1,
            "only `rare`'s single posting drives discovery, not the {ROWS} rows \
             in the candidate union",
        );
        // The read counters reach the same gated surface, and account the
        // postings this corpus holds. They do not show pruning here: `common`'s
        // 40 postings fit in one block, and probing `rare`'s single candidate
        // decodes it whole. `test_combined_fields_read_pruning_engages` is the
        // case sized to span blocks.
        assert!(stats.postings_total >= ROWS as u64, "{stats:?}");
        assert!(stats.blocks_read <= stats.blocks_total, "{stats:?}");
        assert!(stats.postings_read <= stats.postings_total, "{stats:?}");

        // The delegating entry point must return exactly this, stats aside.
        let plain = combined_fields_search(
            &columns,
            &tokens,
            &params,
            Operator::Or,
            &scorer,
            Arc::new(NoFilter),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
        assert_eq!(plain, (row_ids, scores));
    }

    /// One legacy element-per-document column and one modern row-per-document
    /// column in the same query. `docCount'` and `docFreq'` are a `max_f` across
    /// columns, so a column left at element granularity poisons the blend even
    /// when the other column is fine. The whole query must score as if the legacy
    /// column had been reindexed per row.
    #[tokio::test]
    async fn test_combined_fields_mixes_legacy_and_modern_columns() {
        let vocab = ["alpha", "beta"];
        let legacy_rows: ElementRows = vec![
            vec![vec!["alpha"]; 6],
            vec![vec!["beta"]],
            vec![vec!["beta"]],
            vec![vec!["beta"]],
        ];
        let modern_rows: ElementRows = (0..4).map(|_| vec![vec!["alpha", "beta"]]).collect();

        let (legacy, _legacy_dir) =
            element_document_index(InvertedListFormatVersion::V2, &vocab, &legacy_rows).await;
        let (modern, _modern_dir) =
            element_document_index(InvertedListFormatVersion::V3, &vocab, &modern_rows).await;

        let terms = ["alpha".to_owned(), "beta".to_owned()];
        assert_eq!(
            legacy.bm25_stats_for_terms(&terms, None).await.unwrap(),
            (9, 9, vec![6, 3]),
            "the fixture must still be element-per-document",
        );
        assert_eq!(
            legacy.bm25_row_stats_for_terms(&terms, None).await.unwrap(),
            (9, 4, vec![1, 3]),
        );
        assert_eq!(
            modern.bm25_row_stats_for_terms(&terms, None).await.unwrap(),
            modern.bm25_stats_for_terms(&terms, None).await.unwrap(),
            "a row-per-document index must be untouched",
        );

        let columns = combined_columns(vec![legacy, modern.clone()]);
        let mixed = combined_top_k(&columns, &["alpha", "beta"], 4).await;
        // `docCount'` is the 4 rows, not the legacy column's 9 elements.
        let scorer = build_combined_bm25_scorer(
            &columns,
            &Tokens::new(vec!["alpha".to_owned(), "beta".to_owned()], DocType::Text),
            CombinedCorpusStats::IndexOnly,
            None,
        )
        .await
        .unwrap();
        assert_eq!(scorer.doc_count(), 4);

        let (rebuilt_legacy, _rebuilt_dir) = element_document_index(
            InvertedListFormatVersion::V3,
            &vocab,
            &as_row_documents(&legacy_rows),
        )
        .await;
        let rebuilt = combined_top_k(
            &combined_columns(vec![rebuilt_legacy, modern]),
            &["alpha", "beta"],
            4,
        )
        .await;
        assert_eq!(mixed.len(), 4);
        assert_eq!(
            mixed
                .iter()
                .map(|(row_id, score)| (*row_id, score.to_bits()))
                .collect::<Vec<_>>(),
            rebuilt
                .iter()
                .map(|(row_id, score)| (*row_id, score.to_bits()))
                .collect::<Vec<_>>(),
        );
    }
    /// Read pruning has to engage on a real index, not merely be implemented: a
    /// gate that never fires leaves every score correct and silently disables
    /// the pruning, and both arms return the same top-k, so read volume is the
    /// only thing that tells them apart.
    ///
    /// `beta` spans several posting blocks while three `zeta` documents drive
    /// discovery, so at `limit = 1` the threshold rises early, `beta` goes
    /// non-essential, and most of its blocks are never decoded.
    ///
    /// The second case is the counterpart: one row holding two documents makes
    /// the partition's row addresses non-ascending, which rejects the fast path,
    /// so the fallback reads exactly the loads it was handed. Same corpus
    /// otherwise, so the layout gate is the only variable.
    #[rstest]
    #[case::ascending_addresses_prune(false, true)]
    #[case::duplicate_row_reads_everything(true, false)]
    #[tokio::test]
    async fn test_combined_fields_read_pruning_engages(
        #[case] duplicate_row: bool,
        #[case] expect_pruning: bool,
    ) {
        let format_version = InvertedListFormatVersion::V3;
        // Sized in blocks of the widest posting layout, so `beta` spans several
        // of them whichever block size the fixture is written with.
        const NUM_DOCS: usize = 16 * MAX_POSTING_BLOCK_SIZE;
        let has_zeta = |doc: usize| [3, NUM_DOCS / 2, NUM_DOCS - 5].contains(&doc);
        let has_beta = |doc: usize| doc % 5 >= 2;

        // One element per row, so every partition is fast-path eligible on the
        // modern format. `gamma`/`delta` are filler: no query reads them, they
        // only vary `dl'`.
        let mut titles: ElementRows = (0..NUM_DOCS)
            .map(|doc| {
                let mut tokens = vec!["gamma"];
                if has_zeta(doc) {
                    tokens.push("zeta");
                }
                vec![tokens]
            })
            .collect();
        if duplicate_row {
            titles[0].push(vec!["gamma"]);
        }
        let bodies: ElementRows = (0..NUM_DOCS)
            .map(|doc| {
                vec![if has_beta(doc) {
                    vec!["beta"; 1 + doc % 3]
                } else {
                    vec!["delta"]
                }]
            })
            .collect();

        // Derive the corpus shape from the same predicates, so it cannot drift
        // away from what makes the pruning observable.
        let count = |carries: &dyn Fn(usize) -> bool| (0..NUM_DOCS).filter(|d| carries(*d)).count();
        let beta_docs = count(&has_beta);
        let zeta_docs = count(&has_zeta);
        assert!(
            beta_docs > 4 * MAX_POSTING_BLOCK_SIZE && beta_docs < NUM_DOCS,
            "beta must span several blocks without covering the corpus, got {beta_docs} of {NUM_DOCS}"
        );
        assert!(
            zeta_docs < beta_docs / MAX_POSTING_BLOCK_SIZE,
            "zeta ({zeta_docs} docs) must stay under beta's block count ({})",
            beta_docs / MAX_POSTING_BLOCK_SIZE
        );

        let vocab = ["gamma", "zeta", "beta", "delta"];
        let (title_index, _title_dir) =
            element_document_index(format_version, &vocab, &titles).await;
        let (body_index, _body_dir) = element_document_index(format_version, &vocab, &bodies).await;
        // Precondition: the layout gate really is in the state this case wants.
        assert_eq!(
            title_index.partitions[0]
                .docs
                .address_keyed()
                .await
                .unwrap()
                .addresses_strictly_ascending(),
            !duplicate_row,
            "the fixture must set up the layout gate it claims to"
        );
        let columns = combined_columns(vec![title_index, body_index]);

        let (hits, stats) = combined_top_k_with_stats(&columns, &["zeta", "beta"], 1).await;
        assert_eq!(
            hits.len(),
            1,
            "the query must match something to read at all"
        );
        assert!(
            stats.blocks_total > 0 && stats.postings_total > 0,
            "the scan loaded no postings: {stats:?}"
        );
        if expect_pruning {
            assert!(
                2 * stats.blocks_read <= stats.blocks_total,
                "expected block skipping, decoded {} of {} blocks",
                stats.blocks_read,
                stats.blocks_total,
            );
            assert!(
                stats.postings_read < stats.postings_total,
                "expected posting pruning, decoded {} of {} postings",
                stats.postings_read,
                stats.postings_total,
            );
        } else {
            assert_eq!(
                (stats.blocks_read, stats.postings_read),
                (stats.blocks_total, stats.postings_total),
                "the fallback reads exactly the loads it was handed",
            );
        }
    }
}
