//! CPU contracts for the bounded W8 speculative tree. No model or GPU work.

pub(crate) const TREE_W8_ENV: &str = "CAMELID_GEMMA4_MTP12_TREE_W8";

pub(crate) fn enabled(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "{TREE_W8_ENV} must be exactly 0 or 1; got {other:?}"
        )),
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn round_eligible(
    logical: usize,
    physical: usize,
    base: usize,
    capacity: usize,
) -> bool {
    logical == 8 && physical == 8 && base.checked_add(8).is_some_and(|end| end <= capacity)
}

/// Token-aware validation complements the Metal plan's geometry validation.
/// Each node is one target input, rooted at the already-proved anchor.
pub(crate) fn validate(tokens: &[u32], parents: &[i32], depths: &[u32], vocab: usize) -> bool {
    if tokens.len() != 8
        || parents.len() != 8
        || depths.len() != 8
        || parents[0] != -1
        || depths[0] != 0
        || tokens.iter().any(|&token| token as usize >= vocab)
    {
        return false;
    }
    for i in 1..8 {
        let Ok(parent) = usize::try_from(parents[i]) else {
            return false;
        };
        if parent >= i || depths[i] != depths[parent] + 1 {
            return false;
        }
        // A target ID can identify only one child. Equal IDs on different
        // branches are fine because they have different recurrent histories.
        if (1..i).any(|j| parents[j] == parents[i] && tokens[j] == tokens[i]) {
            return false;
        }
    }
    true
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Decision {
    pub path: Vec<usize>,
    pub stop_token: Option<u32>,
    pub next_anchor_token: Option<u32>,
    /// Node whose target prediction has no represented child, if nonterminal.
    pub mismatch_parent: Option<usize>,
}

/// Follow only target-authoritative edges. Stop IDs are never forwarded or
/// emitted, even when the tree contains an exactly matching stop child.
pub(crate) fn accept(
    tokens: &[u32],
    parents: &[i32],
    depths: &[u32],
    target: &[u32],
    stop_ids: &[u32],
    vocab: usize,
) -> Option<Decision> {
    if !validate(tokens, parents, depths, vocab)
        || target.len() != 8
        || target.iter().any(|&id| id as usize >= vocab)
        || stop_ids.contains(&tokens[0])
    {
        return None;
    }
    let mut path = vec![0];
    loop {
        let parent = *path.last()?;
        let next = target[parent];
        if stop_ids.contains(&next) {
            return Some(Decision {
                path,
                stop_token: Some(next),
                next_anchor_token: None,
                mismatch_parent: None,
            });
        }
        let child = (parent + 1..8).find(|&i| parents[i] == parent as i32 && tokens[i] == next);
        if let Some(child) = child {
            path.push(child);
        } else {
            let has_children = parents.contains(&(parent as i32));
            return Some(Decision {
                path,
                stop_token: None,
                next_anchor_token: Some(next),
                mismatch_parent: has_children.then_some(parent),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TOKENS: [u32; 8] = [10, 11, 12, 13, 14, 22, 23, 24];
    const PARENTS: [i32; 8] = [-1, 0, 1, 2, 3, 1, 5, 6];
    const DEPTHS: [u32; 8] = [0, 1, 2, 3, 4, 2, 3, 4];

    #[test]
    fn default_off_strict_selector() {
        assert_eq!(enabled(None), Ok(false));
        assert_eq!(enabled(Some("0")), Ok(false));
        assert_eq!(enabled(Some("1")), Ok(true));
        for s in ["", "true", "01", " 1", "1 "] {
            assert!(enabled(Some(s)).is_err());
        }
    }

    #[test]
    fn follows_alternative_and_uses_its_own_bonus() {
        let target = [11, 22, 13, 14, 98, 23, 24, 25];
        let d = accept(&TOKENS, &PARENTS, &DEPTHS, &target, &[99], 100).unwrap();
        assert_eq!(d.path, [0, 1, 5, 6, 7]);
        assert_eq!(d.next_anchor_token, Some(25));
        assert_eq!(d.mismatch_parent, None);
    }

    #[test]
    fn target_stop_is_never_a_committed_input() {
        for stop_parent in [0, 1, 5, 6, 7] {
            let mut target = [11, 22, 13, 14, 98, 23, 24, 25];
            target[stop_parent] = 99;
            let d = accept(&TOKENS, &PARENTS, &DEPTHS, &target, &[99], 100).unwrap();
            assert_eq!(d.path.last(), Some(&stop_parent));
            assert_eq!(d.stop_token, Some(99));
            assert_eq!(d.next_anchor_token, None);
        }
        let mut tokens = TOKENS;
        tokens[5] = 99;
        let d = accept(
            &tokens,
            &PARENTS,
            &DEPTHS,
            &[11, 99, 13, 14, 98, 23, 24, 25],
            &[99],
            100,
        )
        .unwrap();
        assert_eq!(d.path, [0, 1]);
    }

    #[test]
    fn wrong_branch_predictions_cannot_affect_accepted_path() {
        for poison in 0..100 {
            let target = [11, 22, poison, poison, poison, 23, 24, 25];
            let d = accept(&TOKENS, &PARENTS, &DEPTHS, &target, &[99], 100).unwrap();
            assert_eq!(d.path, [0, 1, 5, 6, 7]);
            assert_eq!(d.next_anchor_token, Some(25));
        }
    }

    #[test]
    fn rejects_ambiguous_siblings_invalid_topology_and_vocabulary() {
        assert!(validate(&TOKENS, &PARENTS, &DEPTHS, 100));
        let mut tokens = TOKENS;
        tokens[5] = tokens[2];
        assert!(!validate(&tokens, &PARENTS, &DEPTHS, 100));
        for parent in [-1, 5, 7, i32::MAX] {
            let mut parents = PARENTS;
            parents[5] = parent;
            assert!(!validate(&TOKENS, &parents, &DEPTHS, 100));
        }
        let mut depths = DEPTHS;
        depths[5] = 5;
        assert!(!validate(&TOKENS, &PARENTS, &depths, 100));
        assert!(!validate(&TOKENS, &PARENTS, &DEPTHS, 24));
    }

    #[test]
    fn every_edge_prefix_has_correct_bonus_or_stop_and_compaction_order() {
        // Exhaust all authoritative paths on this tree, including immediate
        // rejection and a stop either present or absent among the children.
        for leaf in 0..8 {
            let mut path = vec![leaf];
            while *path.last().unwrap() != 0 {
                path.push(PARENTS[*path.last().unwrap()] as usize);
            }
            path.reverse();
            for terminal in [98, 99] {
                let mut target = [97; 8];
                for pair in path.windows(2) {
                    target[pair[0]] = TOKENS[pair[1]];
                }
                target[leaf] = terminal;
                let d = accept(&TOKENS, &PARENTS, &DEPTHS, &target, &[99], 100).unwrap();
                assert_eq!(d.path, path);
                assert_eq!(d.stop_token, (terminal == 99).then_some(terminal));
                assert_eq!(d.next_anchor_token, (terminal != 99).then_some(terminal));
                // Simulate gather-then-scatter; prefix and unused tail intact.
                let mut kv = (0..20).collect::<Vec<_>>();
                let saved = path.iter().map(|&row| kv[10 + row]).collect::<Vec<_>>();
                kv[10..10 + path.len()].copy_from_slice(&saved);
                assert_eq!(&kv[..10], &(0..10).collect::<Vec<_>>());
                assert_eq!(&kv[10..10 + path.len()], saved);
                assert!(path.windows(2).all(|pair| pair[0] < pair[1]));
            }
        }
    }
}
