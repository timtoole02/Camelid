//! Line diff for the divergence view.
//!
//! Written here rather than pulled in as a dependency: the whole feature is
//! about being able to say exactly how two answers differ, and a diff is small
//! enough that owning it is cheaper than trusting it.
//!
//! Standard Myers-style LCS over lines. Two behaviours are not obvious. Lines
//! keep their terminators, because the verdict beside a diff is a hash of the
//! raw bytes and a diff that could not see a missing final newline would call
//! two answers divergent and then show nothing changed. And past a size bound
//! we stop diffing and say so rather than locking up a UI thread: LCS is
//! O(n*m), and two runaway generations could each be tens of thousands of
//! lines. A refused diff is a stated outcome, never a silently empty one.

use serde::Serialize;

/// Above this many lines on either side we decline rather than run an O(n*m)
/// table. Two 2000-line answers already means a 4M-cell table.
const MAX_DIFF_LINES: usize = 2000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Same,
    /// Present on the left side only.
    Removed,
    /// Present on the right side only.
    Added,
}

/// How a line ended.
///
/// Carried beside the text rather than inside it, so a renderer never has to
/// find the terminator itself — and so two lines that differ only here are two
/// different lines rather than one unchanged one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Eol {
    #[serde(rename = "lf")]
    Lf,
    #[serde(rename = "crlf")]
    Crlf,
    /// The answer ended without one, so this is only ever on a final line.
    #[serde(rename = "none")]
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiffLine {
    pub op: Op,
    /// The line without its terminator; see `eol`.
    pub text: String,
    pub eol: Eol,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Diff {
    /// The two sides are byte-identical; there is nothing to render.
    Identical,
    Lines {
        lines: Vec<DiffLine>,
    },
    /// Too large to diff. Carries the reason so the UI never has to invent one.
    Declined {
        reason: String,
    },
}

impl Diff {
    /// Lines that differ, for a summary count. Zero for `Identical`, and zero
    /// for `Declined` — which is why `Declined` is its own variant and must not
    /// be rendered as agreement.
    pub fn changed_lines(&self) -> usize {
        match self {
            Self::Lines { lines } => lines.iter().filter(|line| line.op != Op::Same).count(),
            _ => 0,
        }
    }
}

/// One line of an answer: its text, and how it ended.
type Line<'a> = (&'a str, Eol);

/// Split an answer so that the pieces rejoin into exactly the bytes that were
/// hashed.
///
/// `str::lines` is what this replaced, and it is lossy twice over: it drops a
/// final newline and folds `\r\n` into `\n`. Either one let a byte difference
/// the verdict had counted vanish from the diff shown beside it.
fn split_lines(text: &str) -> Vec<Line<'_>> {
    text.split_inclusive('\n')
        .map(|line| {
            if let Some(body) = line.strip_suffix("\r\n") {
                (body, Eol::Crlf)
            } else if let Some(body) = line.strip_suffix('\n') {
                (body, Eol::Lf)
            } else {
                (line, Eol::Missing)
            }
        })
        .collect()
}

/// Diff two answers line by line.
pub fn diff_lines(left: &str, right: &str) -> Diff {
    if left == right {
        return Diff::Identical;
    }
    let left_lines = split_lines(left);
    let right_lines = split_lines(right);

    if left_lines.len() > MAX_DIFF_LINES || right_lines.len() > MAX_DIFF_LINES {
        return Diff::Declined {
            reason: format!(
                "answers are {} and {} lines; this build declines to diff past {MAX_DIFF_LINES} lines a side",
                left_lines.len(),
                right_lines.len()
            ),
        };
    }

    Diff::Lines {
        lines: walk(
            &lcs_table(&left_lines, &right_lines),
            &left_lines,
            &right_lines,
        ),
    }
}

/// `table[i][j]` = length of the longest common subsequence of `left[i..]` and
/// `right[j..]`, built backwards so the walk below can go forwards and emit
/// lines in reading order.
fn lcs_table(left: &[Line<'_>], right: &[Line<'_>]) -> Vec<Vec<usize>> {
    let mut table = vec![vec![0usize; right.len() + 1]; left.len() + 1];
    for i in (0..left.len()).rev() {
        for j in (0..right.len()).rev() {
            table[i][j] = if left[i] == right[j] {
                table[i + 1][j + 1] + 1
            } else {
                table[i + 1][j].max(table[i][j + 1])
            };
        }
    }
    table
}

fn entry(op: Op, (text, eol): Line<'_>) -> DiffLine {
    DiffLine {
        op,
        text: text.to_string(),
        eol,
    }
}

fn walk(table: &[Vec<usize>], left: &[Line<'_>], right: &[Line<'_>]) -> Vec<DiffLine> {
    let mut lines = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < left.len() && j < right.len() {
        if left[i] == right[j] {
            lines.push(entry(Op::Same, left[i]));
            i += 1;
            j += 1;
        } else if table[i + 1][j] >= table[i][j + 1] {
            lines.push(entry(Op::Removed, left[i]));
            i += 1;
        } else {
            lines.push(entry(Op::Added, right[j]));
            j += 1;
        }
    }
    for line in left.iter().skip(i) {
        lines.push(entry(Op::Removed, *line));
    }
    for line in right.iter().skip(j) {
        lines.push(entry(Op::Added, *line));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(diff: &Diff) -> Vec<(Op, &str)> {
        match diff {
            Diff::Lines { lines } => lines.iter().map(|l| (l.op, l.text.as_str())).collect(),
            other => panic!("expected a line diff, got {other:?}"),
        }
    }

    fn with_endings(diff: &Diff) -> Vec<(Op, &str, Eol)> {
        match diff {
            Diff::Lines { lines } => lines
                .iter()
                .map(|l| (l.op, l.text.as_str(), l.eol))
                .collect(),
            other => panic!("expected a line diff, got {other:?}"),
        }
    }

    #[test]
    fn identical_answers_are_not_rendered_as_a_diff_of_nothing() {
        // A caller must be able to tell "the same" from "we produced no lines".
        assert_eq!(diff_lines("same", "same"), Diff::Identical);
        assert_eq!(diff_lines("", ""), Diff::Identical);
    }

    #[test]
    fn the_measured_seven_versus_twelve_divergence_renders_as_one_change_each_way() {
        let diff = diff_lines("12", "7");
        assert_eq!(rendered(&diff), [(Op::Removed, "12"), (Op::Added, "7")]);
        assert_eq!(diff.changed_lines(), 2);
    }

    #[test]
    fn a_shared_prefix_and_suffix_survive_a_change_in_the_middle() {
        let diff = diff_lines("a\nb\nc", "a\nx\nc");
        assert_eq!(
            rendered(&diff),
            [
                (Op::Same, "a"),
                (Op::Removed, "b"),
                (Op::Added, "x"),
                (Op::Same, "c")
            ]
        );
        assert_eq!(diff.changed_lines(), 2);
    }

    #[test]
    fn an_inserted_line_is_an_addition_rather_than_a_rewrite_of_everything_after_it() {
        let diff = diff_lines("a\nc", "a\nb\nc");
        assert_eq!(
            rendered(&diff),
            [(Op::Same, "a"), (Op::Added, "b"), (Op::Same, "c")]
        );
        assert_eq!(diff.changed_lines(), 1);
    }

    #[test]
    fn an_empty_side_is_wholly_removed_or_wholly_added() {
        assert_eq!(
            rendered(&diff_lines("a\nb", "")),
            [(Op::Removed, "a"), (Op::Removed, "b")]
        );
        assert_eq!(
            rendered(&diff_lines("", "a\nb")),
            [(Op::Added, "a"), (Op::Added, "b")]
        );
    }

    #[test]
    fn trailing_whitespace_is_a_difference_because_it_changes_the_hash() {
        // The verdict is computed from a sha256 of the raw bytes, so a diff
        // that "helpfully" trimmed would contradict the verdict beside it.
        let diff = diff_lines("answer", "answer ");
        assert_eq!(
            rendered(&diff),
            [(Op::Removed, "answer"), (Op::Added, "answer ")]
        );
    }

    /// Measured live: `"12\n"` against `"12"` was reported DIVERGENT beside a
    /// diff of one unchanged line reading `12`.
    #[test]
    fn a_missing_final_newline_is_a_changed_line_rather_than_an_unchanged_one() {
        let diff = diff_lines("12\n", "12");
        assert_eq!(
            with_endings(&diff),
            [
                (Op::Removed, "12", Eol::Lf),
                (Op::Added, "12", Eol::Missing)
            ]
        );
        assert_eq!(diff.changed_lines(), 2);
    }

    #[test]
    fn crlf_and_lf_endings_are_different_lines() {
        let diff = diff_lines("a\r\nb", "a\nb");
        assert_eq!(
            with_endings(&diff),
            [
                (Op::Removed, "a", Eol::Crlf),
                (Op::Added, "a", Eol::Lf),
                (Op::Same, "b", Eol::Missing)
            ]
        );
    }

    /// The property the verdict and the diff have to share. Exhaustive over
    /// every string of up to four characters drawn from the three that decide
    /// line structure, which is every way two answers can differ in their
    /// terminators alone.
    #[test]
    fn any_two_answers_whose_bytes_differ_have_a_changed_line() {
        let mut answers = vec![String::new()];
        let mut frontier = vec![String::new()];
        for _ in 0..4 {
            frontier = frontier
                .iter()
                .flat_map(|prefix| ['a', '\r', '\n'].map(|next| format!("{prefix}{next}")))
                .collect();
            answers.extend(frontier.iter().cloned());
        }
        assert_eq!(answers.len(), 121, "the enumeration covers what it says");

        for left in &answers {
            for right in &answers {
                let diff = diff_lines(left, right);
                if left == right {
                    assert_eq!(diff, Diff::Identical);
                } else {
                    assert!(
                        diff.changed_lines() > 0,
                        "{left:?} and {right:?} differ, but the diff shows no change: {diff:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_lines_of_an_answer_rejoin_into_exactly_its_bytes() {
        for answer in ["", "a", "a\n", "a\r\n", "a\r", "\r\n\n", "x\ny\r\nz"] {
            let rejoined: String = split_lines(answer)
                .iter()
                .map(|(text, eol)| {
                    let ending = match eol {
                        Eol::Lf => "\n",
                        Eol::Crlf => "\r\n",
                        Eol::Missing => "",
                    };
                    format!("{text}{ending}")
                })
                .collect();
            assert_eq!(rejoined, answer);
        }
    }

    /// The WebUI reads these three strings. Renaming a variant must not
    /// quietly change what it is sent, and the text must never carry the
    /// terminator the field already names.
    #[test]
    fn every_line_on_the_wire_names_its_ending_and_carries_text_without_it() {
        let wire = serde_json::to_value(diff_lines("a\r\nb\n", "a\nc")).expect("serialises");
        assert_eq!(wire["kind"], "lines");
        let lines = wire["lines"].as_array().expect("lines");
        let endings: Vec<&str> = lines
            .iter()
            .map(|line| line["eol"].as_str().expect("every line names its ending"))
            .collect();
        assert_eq!(endings, ["crlf", "lf", "lf", "none"]);
        for line in lines {
            let text = line["text"].as_str().expect("text");
            assert!(!text.contains('\n') && !text.contains('\r'), "{line}");
        }
    }

    #[test]
    fn an_oversized_answer_is_declined_out_loud_rather_than_returning_no_lines() {
        let big = (0..=MAX_DIFF_LINES)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        match diff_lines(&big, "short") {
            Diff::Declined { reason } => {
                assert!(reason.contains(&MAX_DIFF_LINES.to_string()), "{reason}");
            }
            other => panic!("expected a declined diff, got {other:?}"),
        }
    }

    #[test]
    fn a_declined_diff_reports_no_changed_lines_but_is_not_identical() {
        let big = (0..=MAX_DIFF_LINES)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let diff = diff_lines(&big, "short");
        assert_eq!(diff.changed_lines(), 0);
        assert_ne!(diff, Diff::Identical, "declining to diff is not agreement");
    }
}
