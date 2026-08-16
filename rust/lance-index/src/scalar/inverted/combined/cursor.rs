// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-term cross-column cursors for `combined_fields`: the [`TermCursor`]
//! abstraction, its lazy block-skipping and eager full-read implementations,
//! the loaded posting sources they are built from, and the read accounting.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::Array;
use lance_select::RowAddrMask;

use super::super::documents::AddressKeyedDocuments;
use super::super::encoding::{
    MAX_POSTING_BLOCK_SIZE, decompress_posting_block, decompress_posting_remainder,
};
use super::super::index::{
    CompressedPostingList, PostingList, PostingTailCodec, live_posting_rows,
};
use super::super::scorer::CombinedFieldsBM25Scorer;
use super::maxscore::{CombinedScanStats, term_upper_bound};

/// The MAXSCORE loop's view of one query term. Both the eager [full-read
/// fallback](MaterializedTerm) and the lazy [block-skipping fast path](LazyTerm)
/// implement this, so [`combined_maxscore`](super::maxscore::combined_maxscore) is
/// one algorithm: the paths differ only in how a `tf'` is fetched, which makes the
/// returned top-k identical by construction.
///
/// A term exposes a merged cross-column cursor over the shared row-id space:
/// [`head`](Self::head) is the smallest not-yet-consumed row id, advanced by
/// the essential terms during discovery; [`probe`](Self::probe) reads `tf'` at
/// an arbitrary (non-decreasing) target for the non-essential terms.
pub(super) trait TermCursor {
    /// The constant per-term ceiling `idf · MAX_DOC_WEIGHT` (clamped, see
    /// [`term_upper_bound`]).
    fn upper_bound(&self) -> f32;
    /// The blended IDF `idf'(t)`.
    fn idf(&self) -> f32;
    /// The smallest not-yet-consumed row id across the term's columns, or
    /// `None` when the term is exhausted.
    fn head(&self) -> Option<u64>;
    /// `tf'` at [`head`](Self::head) (the weighted sum across the columns that
    /// carry the head row id). Only meaningful while `head()` is `Some`.
    fn head_tf(&self) -> f32;
    /// Consume `row_id`: advance every column cursor currently sitting on it.
    /// Called only for `row_id == head()`.
    fn consume(&mut self, row_id: u64);
    /// `tf'` at `target`, or 0 when the term is absent there. `target` never
    /// decreases across calls to one cursor, so the lazy path can skip blocks.
    fn probe(&mut self, target: u64) -> f32;
}

/// One query term's postings, merged across every target column/partition into
/// the shared row-id space: the full-read fallback. Used whenever the fast
/// path cannot prove block skipping safe (legacy layout, unsorted `row_ids`,
/// non-compressed postings) and by the MAXSCORE unit tests.
///
/// Entries are unique row ids sorted ascending, each carrying the blended term
/// frequency `tf'(t, d) = Σ_f w_f · freq_f(t, d)`.
pub(super) struct CombinedTermPostings {
    pub(super) idf: f32,
    pub(super) upper_bound: f32,
    pub(super) postings: Vec<(u64, f32)>,
}

impl CombinedTermPostings {
    /// `tf'` for `row_id`, or 0 when the term does not occur in the document.
    #[inline]
    pub(super) fn tf_prime(&self, row_id: u64) -> f32 {
        match self.postings.binary_search_by_key(&row_id, |(id, _)| *id) {
            Ok(idx) => self.postings[idx].1,
            Err(_) => 0.0,
        }
    }
}

/// [`TermCursor`] over a [`CombinedTermPostings`] with an owned scan cursor.
pub(super) struct MaterializedTerm {
    pub(super) postings: CombinedTermPostings,
    cursor: usize,
}

impl MaterializedTerm {
    pub(super) fn new(postings: CombinedTermPostings) -> Self {
        Self {
            postings,
            cursor: 0,
        }
    }
}

impl TermCursor for MaterializedTerm {
    #[inline]
    fn upper_bound(&self) -> f32 {
        self.postings.upper_bound
    }

    #[inline]
    fn idf(&self) -> f32 {
        self.postings.idf
    }

    #[inline]
    fn head(&self) -> Option<u64> {
        self.postings.postings.get(self.cursor).map(|(id, _)| *id)
    }

    #[inline]
    fn head_tf(&self) -> f32 {
        self.postings
            .postings
            .get(self.cursor)
            .map(|(_, tf)| *tf)
            .unwrap_or(0.0)
    }

    #[inline]
    fn consume(&mut self, row_id: u64) {
        if self.head() == Some(row_id) {
            self.cursor += 1;
        }
    }

    #[inline]
    fn probe(&mut self, target: u64) -> f32 {
        self.postings.tf_prime(target)
    }
}

/// One `(column, index, partition)` compressed posting list feeding a term's
/// cross-field cursor on the fast path. Decodes posting blocks lazily and skips
/// (never decodes) the blocks a row-id seek jumps past. Correct only when the
/// partition's `row_ids` are strictly ascending, so doc-id order equals row-id
/// order and a block's row-id span is an interval (checked by the caller via
/// [`AddressKeyedDocuments::addresses_strictly_ascending`]).
pub(super) struct FastPostingSource {
    weight: f32,
    docs: AddressKeyedDocuments,
    mask: Arc<RowAddrMask>,
    list: CompressedPostingList,
    block_size: usize,
    tail_codec: PostingTailCodec,
    num_blocks: usize,
    remainder: usize,
    /// The block currently held in `doc_ids`/`freqs`, or `usize::MAX` if none.
    decoded_block: usize,
    doc_ids: Vec<u32>,
    freqs: Vec<u32>,
    /// Index of the head posting within the decoded block.
    within: usize,
    buffer: Box<[u32; MAX_POSTING_BLOCK_SIZE]>,
    /// Head = first selected posting at or after `within`; `None` when exhausted.
    head_row_id: Option<u64>,
    head_freq: u32,
    // Read accounting (this source only).
    blocks_decoded: u64,
    postings_decoded: u64,
}

impl FastPostingSource {
    pub(super) fn new(
        weight: f32,
        docs: AddressKeyedDocuments,
        mask: Arc<RowAddrMask>,
        list: CompressedPostingList,
    ) -> Self {
        let block_size = list.block_size;
        let num_blocks = list.blocks.len();
        let remainder = list.length as usize % block_size;
        let mut source = Self {
            weight,
            docs,
            mask,
            tail_codec: list.posting_tail_codec,
            list,
            block_size,
            num_blocks,
            remainder,
            decoded_block: usize::MAX,
            doc_ids: Vec::with_capacity(block_size),
            freqs: Vec::with_capacity(block_size),
            within: 0,
            buffer: Box::new([0; MAX_POSTING_BLOCK_SIZE]),
            head_row_id: None,
            head_freq: 0,
            blocks_decoded: 0,
            postings_decoded: 0,
        };
        // Establish the initial head (block 0 is always needed: every term is
        // essential until the top-k heap fills).
        if source.num_blocks > 0 {
            source.decode(0);
            source.scan_forward(0);
        }
        source
    }

    /// Decode block `block_idx` into `doc_ids`/`freqs`, reset `within`, and
    /// count the read. Doc ids come out absolute and ascending.
    fn decode(&mut self, block_idx: usize) {
        let block = self.list.blocks.value(block_idx);
        self.doc_ids.clear();
        self.freqs.clear();
        if block_idx + 1 == self.num_blocks && self.remainder != 0 {
            decompress_posting_remainder(
                block,
                self.remainder,
                self.tail_codec,
                self.block_size,
                &mut self.doc_ids,
                &mut self.freqs,
            );
        } else {
            decompress_posting_block(
                block,
                &mut self.buffer[..],
                &mut self.doc_ids,
                &mut self.freqs,
                self.block_size,
            );
        }
        self.decoded_block = block_idx;
        self.within = 0;
        self.blocks_decoded += 1;
        self.postings_decoded += self.doc_ids.len() as u64;
    }

    /// From the current `(decoded_block, within)`, set the head to the first
    /// mask-selected posting whose row id is at least `min_row_id`, decoding
    /// later blocks as needed. Exhausts the source (head `None`) when none
    /// remains. Pass 0 to settle on the next selected posting wherever it is.
    fn scan_forward(&mut self, min_row_id: u64) {
        loop {
            while self.within < self.doc_ids.len() {
                let row_id = self.docs.row_address(self.doc_ids[self.within]);
                if row_id >= min_row_id && self.mask.selected(row_id) {
                    self.head_row_id = Some(row_id);
                    self.head_freq = self.freqs[self.within];
                    return;
                }
                self.within += 1;
            }
            if self.decoded_block + 1 >= self.num_blocks {
                self.head_row_id = None;
                return;
            }
            self.decode(self.decoded_block + 1);
        }
    }

    /// Advance past the current head to the next selected posting.
    fn advance(&mut self) {
        self.within += 1;
        self.scan_forward(0);
    }

    /// Position the head at the first selected posting with `row_id >= target`,
    /// skipping (not decoding) blocks whose entire row-id span is below
    /// `target`. `target` must not decrease across calls.
    fn seek(&mut self, target: u64) {
        // Skip whole blocks: block `b` is entirely below `target` when the next
        // block's least row id is `<= target` (row ids ascend across blocks, so
        // block `b`'s max row id is strictly below block `b + 1`'s least). Start
        // from the decoded block; monotone `target` never moves us backward.
        let mut block_idx = if self.decoded_block == usize::MAX {
            0
        } else {
            self.decoded_block
        };
        while block_idx + 1 < self.num_blocks {
            let next_least_doc = self.list.block_least_doc_id(block_idx + 1);
            if self.docs.row_address(next_least_doc) <= target {
                block_idx += 1;
            } else {
                break;
            }
        }
        if self.decoded_block != block_idx {
            self.decode(block_idx);
        }
        // Scan forward to the first selected posting at or after `target`,
        // spilling into later blocks if this one ends before `target`.
        self.scan_forward(target);
    }

    #[inline]
    fn contribution(&self) -> f32 {
        self.weight * self.head_freq as f32
    }
}

/// [`TermCursor`] merging a term's [`FastPostingSource`]s across columns in the
/// shared row-id space: the read-pruning fast path.
pub(super) struct LazyTerm {
    idf: f32,
    upper_bound: f32,
    sources: Vec<FastPostingSource>,
    /// The merged head: the least `head_row_id` over `sources`, or `None` once
    /// every source is exhausted. Cached rather than rescanned because
    /// `combined_maxscore` reads it several times per candidate per term, and
    /// refreshed by every operation that moves a source's head.
    head: Option<u64>,
}

impl LazyTerm {
    pub(super) fn new(idf: f32, sources: Vec<FastPostingSource>) -> Self {
        let mut term = Self {
            idf,
            upper_bound: term_upper_bound(idf),
            sources,
            head: None,
        };
        term.refresh_head();
        term
    }

    #[inline]
    fn refresh_head(&mut self) {
        self.head = self.sources.iter().filter_map(|s| s.head_row_id).min();
    }
}

impl TermCursor for LazyTerm {
    #[inline]
    fn upper_bound(&self) -> f32 {
        self.upper_bound
    }

    #[inline]
    fn idf(&self) -> f32 {
        self.idf
    }

    #[inline]
    fn head(&self) -> Option<u64> {
        self.head
    }

    fn head_tf(&self) -> f32 {
        // Sum in source (column → index → partition) order so `tf'` is
        // bit-identical to the eager scan's ordered accumulation.
        let Some(head) = self.head else {
            return 0.0;
        };
        let mut tf = 0.0f32;
        for source in &self.sources {
            if source.head_row_id == Some(head) {
                tf += source.contribution();
            }
        }
        tf
    }

    fn consume(&mut self, row_id: u64) {
        for source in &mut self.sources {
            if source.head_row_id == Some(row_id) {
                source.advance();
            }
        }
        self.refresh_head();
    }

    fn probe(&mut self, target: u64) -> f32 {
        let mut tf = 0.0f32;
        for source in &mut self.sources {
            source.seek(target);
            if source.head_row_id == Some(target) {
                tf += source.contribution();
            }
        }
        // Every source moved to its first posting at or after `target`.
        self.refresh_head();
        tf
    }
}

/// A `(column, index, partition)` posting source loaded for one term, retained
/// so the fast-path/fallback decision (which needs every posting's layout) is
/// made once without re-reading.
pub(super) struct LoadedSource {
    pub(super) weight: f32,
    pub(super) docs: AddressKeyedDocuments,
    pub(super) is_legacy: bool,
    pub(super) posting: PostingList,
}

impl LoadedSource {
    /// The fast-path form of this source, or `None` when its posting list is not
    /// compressed and therefore cannot be block skipped.
    fn into_compressed(self) -> Option<CompressedSource> {
        match self.posting {
            PostingList::Compressed(list) => Some(CompressedSource {
                weight: self.weight,
                docs: self.docs,
                list,
            }),
            PostingList::Plain(_) => None,
        }
    }
}

/// A [`LoadedSource`] that carries fast-path eligibility in its type: the
/// posting list is compressed, so [`FastPostingSource`] can decode it block by
/// block. `is_legacy` is not kept because a legacy partition never reaches here.
pub(super) struct CompressedSource {
    weight: f32,
    docs: AddressKeyedDocuments,
    list: CompressedPostingList,
}

/// Retype every loaded posting as a [`CompressedSource`], or hand the loads back
/// unchanged when any of them is not compressed.
///
/// All or nothing: a [`LazyTerm`] merges one term's sources into a single
/// cursor, so one plain posting anywhere forces the whole query onto the eager
/// fallback, which scans exactly these loads rather than repeating them.
pub(super) fn into_compressed_sources(
    loaded: Vec<Vec<LoadedSource>>,
) -> std::result::Result<Vec<Vec<CompressedSource>>, Vec<Vec<LoadedSource>>> {
    if !loaded
        .iter()
        .flatten()
        .all(|source| matches!(source.posting, PostingList::Compressed(_)))
    {
        return Err(loaded);
    }
    // Lossless because of the check above.
    Ok(loaded
        .into_iter()
        .map(|sources| {
            sources
                .into_iter()
                .filter_map(LoadedSource::into_compressed)
                .collect()
        })
        .collect())
}

/// Build a lazy fast-path cursor for `term` over its already-retyped
/// [`CompressedSource`]s.
pub(super) fn build_lazy_term(
    term: &str,
    sources: Vec<CompressedSource>,
    mask: &Arc<RowAddrMask>,
    scorer: &CombinedFieldsBM25Scorer,
) -> LazyTerm {
    let sources = sources
        .into_iter()
        .map(|source| FastPostingSource::new(source.weight, source.docs, mask.clone(), source.list))
        .collect();
    LazyTerm::new(scorer.query_weight(term), sources)
}

/// Build the eager full-read cursor for `term` by merging every source's
/// postings into the shared row-id space, accumulating `tf'` in the canonical
/// order and accounting the reads (the fallback reads everything).
pub(super) fn build_materialized_term(
    term: &str,
    sources: Vec<LoadedSource>,
    mask: &Arc<RowAddrMask>,
    scorer: &CombinedFieldsBM25Scorer,
    stats: &mut CombinedScanStats,
) -> MaterializedTerm {
    let mut acc: HashMap<u64, f32> = HashMap::new();
    for source in &sources {
        let posting_len = source.posting.len() as u64;
        let blocks = match &source.posting {
            PostingList::Compressed(list) => list.blocks.len() as u64,
            PostingList::Plain(_) => 1,
        };
        stats.postings_total += posting_len;
        stats.postings_read += posting_len;
        stats.blocks_total += blocks;
        stats.blocks_read += blocks;
        for (row_id, freq) in live_posting_rows(&source.posting, &source.docs, source.is_legacy) {
            if !mask.selected(row_id) {
                continue;
            }
            *acc.entry(row_id).or_insert(0.0) += source.weight * freq as f32;
        }
    }
    let idf = scorer.query_weight(term);
    let mut postings: Vec<(u64, f32)> = acc.into_iter().collect();
    postings.sort_unstable_by_key(|(row_id, _)| *row_id);
    MaterializedTerm::new(CombinedTermPostings {
        idf,
        upper_bound: term_upper_bound(idf),
        postings,
    })
}

/// Accumulate the fast-path term cursors' per-source read counters, which only
/// they know: what they decoded is settled once the MAXSCORE loop is done with
/// them.
pub(super) fn record_fast_reads(cursors: &[LazyTerm], stats: &mut CombinedScanStats) {
    for term in cursors {
        for source in &term.sources {
            stats.blocks_total += source.num_blocks as u64;
            stats.blocks_read += source.blocks_decoded;
            stats.postings_total += source.list.length as u64;
            stats.postings_read += source.postings_decoded;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::maxscore::combined_maxscore;
    use super::super::testing::{
        DocsSource, TermSpec, assert_lazy_matches_materialized, compressed_list, identity_docs,
        modern_identity_docs,
    };
    use super::*;
    use lance_core::utils::address::RowAddress;
    use lance_select::RowAddrTreeMap;
    use rstest::rstest;
    use std::cmp::Reverse;

    // Fast-path (lazy, block-skipping) equivalence.
    //
    // These tests build real compressed posting blocks and document views, then
    // assert the lazy cursors return the same top-k as the materialized ones,
    // bit-exact scores and row ids in the same order, and that both match the
    // independent exact oracle. Coverage spans OR/AND, every k, block skipping,
    // absent columns, a masked prefilter, ties, and a non-positive-idf term.
    //
    // Both [`AddressKeyedDocuments`] representations are exercised, covering the
    // lazy cursors' real address and length lookups. Only the modern one is ever
    // paired with the fast path in production, because a legacy partition's
    // posting layout forces the fallback.

    #[rstest]
    #[case::legacy(DocsSource::Legacy)]
    #[case::modern(DocsSource::Modern)]
    #[tokio::test]
    async fn test_lazy_fast_path_matches_materialized_skew(#[case] source: DocsSource) {
        // 500 docs, two columns with varying lengths. A dense common term spans
        // four posting blocks (so block skipping actually fires when it goes
        // non-essential); rare/mid terms with far-apart docs drive discovery and
        // force cross-block seeks into the common term. `body_only` is absent
        // from column 0, `rare` is present in both, `zero_idf` has a clamped
        // (0) ceiling but still contributes its exact score.
        let col0: Vec<u32> = (0..500).map(|d| 3 + (d % 5) as u32).collect();
        let col1: Vec<u32> = (0..500).map(|d| 2 + (d % 7) as u32).collect();
        let common: Vec<(u32, u32)> = (0..500u32).map(|d| (d, 1 + d % 3)).collect();
        let terms = [
            TermSpec {
                idf: 0.05,
                columns: vec![common.clone(), common],
            },
            TermSpec {
                idf: 6.0,
                columns: vec![vec![(5, 2), (250, 1), (495, 3)], vec![(5, 1)]],
            },
            TermSpec {
                idf: 1.5,
                columns: vec![vec![(10, 1), (300, 2)], vec![(10, 1), (260, 1)]],
            },
            TermSpec {
                idf: 3.0,
                columns: vec![vec![], vec![(3, 2), (400, 1)]],
            },
            TermSpec {
                idf: 0.0,
                columns: vec![vec![(7, 4), (8, 1)], vec![(7, 1)]],
            },
        ];
        assert_lazy_matches_materialized(
            source,
            &[col0, col1],
            &[2.0, 1.0],
            &terms,
            15.0,
            Arc::new(RowAddrMask::all_rows()),
            "skew",
        )
        .await;
    }

    #[rstest]
    #[case::legacy(DocsSource::Legacy)]
    #[case::modern(DocsSource::Modern)]
    #[tokio::test]
    async fn test_lazy_fast_path_matches_materialized_masked(#[case] source: DocsSource) {
        // Same corpus but a prefilter blocks a handful of rows, including some a
        // seek would otherwise land on; the lazy cursor must skip them exactly
        // as the eager merge drops them.
        let col0: Vec<u32> = (0..500).map(|d| 3 + (d % 5) as u32).collect();
        let col1: Vec<u32> = (0..500).map(|d| 2 + (d % 7) as u32).collect();
        let common: Vec<(u32, u32)> = (0..500u32).map(|d| (d, 1)).collect();
        let terms = [
            TermSpec {
                idf: 0.05,
                columns: vec![common.clone(), common],
            },
            TermSpec {
                idf: 6.0,
                columns: vec![vec![(5, 2), (250, 1), (495, 3)], vec![(5, 1), (250, 2)]],
            },
        ];
        let blocked = RowAddrTreeMap::from_iter([5u64, 10, 128, 250, 400]);
        let mask = RowAddrMask::all_rows().also_block(blocked);
        assert_lazy_matches_materialized(
            source,
            &[col0, col1],
            &[2.0, 1.0],
            &terms,
            15.0,
            Arc::new(mask),
            "masked",
        )
        .await;
    }

    #[rstest]
    #[case::legacy(DocsSource::Legacy)]
    #[case::modern(DocsSource::Modern)]
    #[tokio::test]
    async fn test_lazy_fast_path_matches_materialized_ties(#[case] source: DocsSource) {
        // Constant lengths + uniform frequency make every matching doc score
        // identically. Both cursors drive the collector with the same candidate
        // order and bit-identical scores, so they must keep the same k docs, not
        // an arbitrary subset each.
        let col0: Vec<u32> = vec![4; 40];
        let col1: Vec<u32> = vec![3; 40];
        let all: Vec<(u32, u32)> = (0..40u32).map(|d| (d, 1)).collect();
        let terms = [TermSpec {
            idf: 2.0,
            columns: vec![all, vec![]],
        }];
        assert_lazy_matches_materialized(
            source,
            &[col0, col1],
            &[1.0, 1.0],
            &terms,
            7.0,
            Arc::new(RowAddrMask::all_rows()),
            "ties",
        )
        .await;
    }

    #[tokio::test]
    async fn test_build_materialized_term_merges_legacy_and_compressed() {
        // Fallback data path: a legacy (Plain, row-id-keyed, list-multiplicity)
        // source and a compressed source merge into one ordered `tf'` stream,
        // masked rows dropped, contributions summed in column order, and the
        // read counters report the full read (no pruning on the fallback).
        use super::super::super::index::PlainPostingList;
        use arrow::buffer::ScalarBuffer;

        let scorer = CombinedFieldsBM25Scorer::new(1000, 12.0, HashMap::new());
        // Legacy column (weight 2): the posting keys directly on row ids; row 20
        // appears twice (list multiplicity), so its contributions sum.
        let legacy = PostingList::Plain(PlainPostingList::new(
            ScalarBuffer::from(vec![10u64, 20, 20, 30]),
            ScalarBuffer::from(vec![1.0f32, 2.0, 3.0, 1.0]),
            Some(0.0),
            None,
        ));
        // Compressed column (weight 1): doc id 20 maps through the modern
        // projection (the only representation a compressed posting is loaded
        // alongside) to row 20; row 42 is blocked by the mask below.
        let compressed = PostingList::Compressed(compressed_list(&[(20, 5), (42, 7)]));
        let docs = identity_docs(DocsSource::Modern, &vec![1u32; 64]).await;
        let sources = vec![
            LoadedSource {
                weight: 2.0,
                docs: docs.clone(),
                is_legacy: true,
                posting: legacy,
            },
            LoadedSource {
                weight: 1.0,
                docs,
                is_legacy: false,
                posting: compressed,
            },
        ];
        let mask = Arc::new(RowAddrMask::all_rows().also_block(RowAddrTreeMap::from_iter([42u64])));
        let mut stats = CombinedScanStats::default();
        let term = build_materialized_term("t", sources, &mask, &scorer, &mut stats);

        // row 10: 2*1 = 2; row 20: 2*2 + 2*3 + 1*5 = 15; row 30: 2*1 = 2.
        // Row 42 is masked out entirely.
        assert_eq!(
            term.postings.postings,
            vec![(10, 2.0), (20, 15.0), (30, 2.0)]
        );
        // Fallback reads everything it loaded: 4 plain + 2 compressed postings.
        assert_eq!(stats.postings_total, 6);
        assert_eq!(stats.postings_read, stats.postings_total);
        assert_eq!(stats.blocks_read, stats.blocks_total);
    }

    #[tokio::test]
    async fn test_build_materialized_term_skips_tombstoned_addresses() {
        // A remapped partition keeps a deleted document's DocId slot so the
        // posting lists stay aligned and answers `TOMBSTONE_ROW` for its address.
        // Nothing else stops that address: a default mask is an empty block list,
        // which selects it, and `doc_length_at(TOMBSTONE_ROW) == 0` would give it
        // the largest `doc_weight` there is. It stays out of the result today only
        // because `TOMBSTONE_ROW == u64::MAX` collides with `combined_maxscore`'s
        // exhausted-cursor sentinel, so it must be dropped at the source.
        const DEAD_ROW: u64 = 20;
        let scorer = CombinedFieldsBM25Scorer::new(1000, 12.0, HashMap::new());
        let docs = modern_identity_docs(&[4u32; 40], &[DEAD_ROW]).await;
        assert_eq!(
            docs.row_address(DEAD_ROW as u32),
            RowAddress::TOMBSTONE_ROW,
            "the deleted document must keep its slot as a tombstone"
        );
        assert_eq!(docs.doc_length_at(RowAddress::TOMBSTONE_ROW), 0);
        // The dead slot also sends this partition down the fallback path that
        // `build_materialized_term` serves.
        assert!(
            !docs.addresses_strictly_ascending(),
            "a tombstoned slot must reject the fast path"
        );

        let sources = vec![LoadedSource {
            weight: 2.0,
            docs: docs.clone(),
            is_legacy: false,
            posting: PostingList::Compressed(compressed_list(&[
                (10, 1),
                (DEAD_ROW as u32, 7),
                (30, 3),
            ])),
        }];
        let mut stats = CombinedScanStats::default();
        let term = build_materialized_term(
            "t",
            sources,
            &Arc::new(RowAddrMask::default()),
            &scorer,
            &mut stats,
        );
        assert_eq!(
            term.postings.postings,
            vec![(10, 2.0), (30, 6.0)],
            "the tombstoned address must never be accumulated, and the live rows \
             must keep their exact contributions"
        );

        // Nor may it be collected: the two live rows are the whole top-k, scored
        // exactly as if the dead slot's posting did not exist.
        //
        // `test_scorer` carries no document frequencies, so `query_weight` is 0 for
        // every term and both rows score 0. The order is therefore the tiebreak,
        // ascending by row id.
        let mut cursors = vec![term];
        let dl_prime = |row_id: u64| -> f32 { 2.0 * docs.doc_length_at(row_id) as f32 };
        let top = combined_maxscore(&mut cursors, dl_prime, 10, false, &scorer, &mut stats);
        let hits: Vec<(u64, u32)> = top
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(doc)| (doc.row_id.0, doc.score.0.to_bits()))
            .collect();
        let expected = |tf: f32| (scorer.query_weight("t") * scorer.doc_weight(tf, 8.0)).to_bits();
        assert_eq!(hits, vec![(10, expected(2.0)), (30, expected(6.0))]);
    }
}
