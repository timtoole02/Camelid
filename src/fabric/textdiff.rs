//! Line diff for the divergence view.
//!
//! Written here rather than pulled in as a dependency: the whole feature is
//! about being able to say exactly how two answers differ, and a diff is small
//! enough that owning it is cheaper than trusting it.
//!
//! Standard Myers-style LCS over lines. The only non-obvious behaviour is the
//! size cap: LCS is O(n*m), and two runaway generations could each be tens of
//! thousands of lines, so past a bound we stop diffing and say so rather than
//! locking up a UI thread. A refused diff is a stated outcome, never a silently
//! empty one.

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiffLine {
    pub op: Op,
    pub text: String,
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

/// Diff two answers line by line.
pub fn diff_lines(left: &str, right: &str) -> Diff {
    if left == right {
        return Diff::Identical;
    }
    let left_lines: Vec<&str> = left.lines().collect();
    let right_lines: Vec<&str> = right.lines().collect();

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
fn lcs_table(left: &[&str], right: &[&str]) -> Vec<Vec<usize>> {
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

fn walk(table: &[Vec<usize>], left: &[&str], right: &[&str]) -> Vec<DiffLine> {
    let mut lines = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < left.len() && j < right.len() {
        if left[i] == right[j] {
            lines.push(DiffLine {
                op: Op::Same,
                text: left[i].to_string(),
            });
            i += 1;
            j += 1;
        } else if table[i + 1][j] >= table[i][j + 1] {
            lines.push(DiffLine {
                op: Op::Removed,
                text: left[i].to_string(),
            });
            i += 1;
        } else {
            lines.push(DiffLine {
                op: Op::Added,
                text: right[j].to_string(),
            });
            j += 1;
        }
    }
    for line in left.iter().skip(i) {
        lines.push(DiffLine {
            op: Op::Removed,
            text: (*line).to_string(),
        });
    }
    for line in right.iter().skip(j) {
        lines.push(DiffLine {
            op: Op::Added,
            text: (*line).to_string(),
        });
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
