#!/usr/bin/env bash
# combined_fields (BM25F) validation: Lance persistent scanner vs Apache Lucene.
#
# Generates a shared two-field corpus, runs Lance's CombinedFieldsQuery through
# the real Dataset/InvertedIndex scanner, runs Lucene's CombinedFieldQuery over
# the same corpus, and reports:
#   - recall@k of Lance vs an exact brute-force BM25F oracle,
#   - recall@k of Lucene vs the same oracle,
#   - Lance<->Lucene mutual top-k overlap (Jaccard).
# Rankings are compared, not absolute scores (Lance keeps a constant (k1+1)
# numerator Lucene lacks; Lucene quantizes norms). Gate: all metrics >= MIN_OK.
#
# Usage: rust/lance/benches/fts/run_combined_fields_compare.sh
#
# Env:
#   DOCS, VOCAB, QUERIES, K   corpus/query knobs (defaults 1000/40/40/10)
#   MIN_OK                    pass threshold (default 0.95)
#   LUCENE_DIR                Lucene source checkout (default ~/repos/extern/lucene)
#   LUCENE_CP                 prebuilt Lucene classpath (skips the gradle build)
#   JAVA_HOME                 JDK 21+ home (falls back to `java` on PATH)
#   WORK                      scratch dir. Must be absent or empty.
#                             Default: temp dir, removed on success.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)" || exit 1
[ -n "$REPO_ROOT" ] || { echo "ERROR: could not resolve repo root" >&2; exit 1; }
cd "$REPO_ROOT" || exit 1

DOCS="${DOCS:-1000}"
VOCAB="${VOCAB:-40}"
QUERIES="${QUERIES:-40}"
K="${K:-10}"
MIN_OK="${MIN_OK:-0.95}"
LUCENE_DIR="${LUCENE_DIR:-$HOME/repos/extern/lucene}"

WORK="${WORK:-}"
if [ -e "$WORK" ]; then
    [ -d "$WORK" ] || { echo "ERROR: WORK is not a directory: $WORK" >&2; exit 1; }
    [ -z "$(ls -A "$WORK" 2>/dev/null)" ] || {
        echo "ERROR: WORK is not empty: $WORK" >&2
        echo "       pass an empty or nonexistent dir, or unset WORK for a temp dir" >&2
        exit 1
    }
fi

# ---- resolve a working cargo (the ~/.cargo/bin shim can be broken) ----
if ! cargo --version >/dev/null 2>&1; then
    CARGO_BIN="$(rustup which cargo 2>/dev/null || true)"
    [ -n "$CARGO_BIN" ] && export PATH="$(dirname "$CARGO_BIN"):$PATH"
fi
cargo --version >/dev/null 2>&1 || { echo "ERROR: cargo not found" >&2; exit 1; }

# ---- locate a JDK (21+) ----
JAVA="${JAVA_HOME:+$JAVA_HOME/bin/java}"; JAVA="${JAVA:-java}"
JAVAC="${JAVA_HOME:+$JAVA_HOME/bin/javac}"; JAVAC="${JAVAC:-javac}"
"$JAVA" -version >/dev/null 2>&1 || { echo "ERROR: java not found" >&2; exit 1; }
echo "JDK: $("$JAVA" -version 2>&1 | head -1)"

# ---- build Lucene classpath ----
if [ -z "${LUCENE_CP:-}" ]; then
    [ -x "$LUCENE_DIR/gradlew" ] || { echo "ERROR: set LUCENE_CP or a valid LUCENE_DIR" >&2; exit 1; }
    CORE_JAR="$(find "$LUCENE_DIR/lucene/core/build/libs" -name 'lucene-core-*.jar' 2>/dev/null | head -1)"
    if [ -z "$CORE_JAR" ]; then
        echo "=== Building Lucene jars ($LUCENE_DIR) ==="
        ( cd "$LUCENE_DIR" && ./gradlew -q :lucene:core:jar :lucene:analysis:common:jar ) \
            || { echo "ERROR: Lucene build failed" >&2; exit 1; }
        CORE_JAR="$(find "$LUCENE_DIR/lucene/core/build/libs" -name 'lucene-core-*.jar' | head -1)"
    fi
    ANALYSIS_JAR="$(find "$LUCENE_DIR/lucene/analysis/common/build/libs" -name 'lucene-analysis-common-*.jar' | head -1)"
    LUCENE_CP="$CORE_JAR:$ANALYSIS_JAR"
fi
echo "Lucene classpath: $LUCENE_CP"

# ---- create the scratch dir (all preconditions hold by now) ----
if [ -n "$WORK" ]; then
    mkdir -p "$WORK" || exit 1
else
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/combined_fields_compare.XXXXXX")" || exit 1
    # Safe to remove: only this run can name it. Kept on failure for inspection.
    cleanup_work() {
        local rc=$?
        if [ "$rc" -eq 0 ]; then rm -rf "$WORK"; else echo "work dir kept: $WORK" >&2; fi
    }
    trap cleanup_work EXIT
fi
echo "work dir: $WORK"

# ---- build + run the Lance side ----
echo "=== Building Lance combined_fields bench ==="
rm -f "$REPO_ROOT"/target/release/deps/combined_fields_compare-*
cargo bench -p lance --bench combined_fields_compare --no-run
LANCE_BIN="$(find "$REPO_ROOT/target/release/deps" -maxdepth 1 -type f -perm -111 \
    -name 'combined_fields_compare-*' ! -name '*.d' -exec ls -t {} + | head -1)"
echo "Lance bench: $LANCE_BIN"
echo "=== Generating corpus + running Lance ==="
"$LANCE_BIN" --out-dir "$WORK" --docs "$DOCS" --vocab "$VOCAB" --queries "$QUERIES" --k "$K"

# ---- compile + run the Lucene side ----
echo "=== Compiling + running Lucene ==="
"$JAVAC" -cp "$LUCENE_CP" -d "$WORK" "$SCRIPT_DIR/LuceneCombinedFieldsBench.java" \
    || { echo "ERROR: javac failed" >&2; exit 1; }
"$JAVA" -cp "$LUCENE_CP:$WORK" LuceneCombinedFieldsBench --in-dir "$WORK"

# ---- compare ----
echo "=== Results ==="
python3 - "$WORK" "$K" "$MIN_OK" <<'PY'
import sys
work, k, min_ok = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])

def rows(name):
    return [[t for t in line.split()] for line in open(f"{work}/{name}").read().splitlines()]

lance, lucene, truth = rows("lance_topk.txt"), rows("lucene_topk.txt"), rows("truth.txt")
n = min(len(lance), len(lucene), len(truth))

def recall(pred):
    tot = 0.0
    for i in range(n):
        t = set(truth[i])
        if not t:
            continue
        tot += len(set(pred[i]) & t) / len(t)
    return tot / n if n else float("nan")

def overlap(a, b):
    tot = 0.0
    for i in range(n):
        sa, sb = set(a[i]), set(b[i])
        tot += len(sa & sb) / max(len(sa | sb), 1)
    return tot / n if n else float("nan")

lance_recall = recall(lance)
lucene_recall = recall(lucene)
mutual = overlap(lance, lucene)
print(f"  lance  recall@{k} vs brute-force BM25F = {lance_recall:.4f}")
print(f"  lucene recall@{k} vs brute-force BM25F = {lucene_recall:.4f}")
print(f"  lance <-> lucene mutual top-{k} overlap = {mutual:.4f}")
ok = min(lance_recall, lucene_recall, mutual) >= min_ok
print(f"  gate (>= {min_ok}): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
