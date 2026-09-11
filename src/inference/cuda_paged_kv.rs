use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const CUDA_KV_PAGE_TOKENS: usize = 16;

static CUDA_KV_PAGE_ALLOCATION_FAILURES: AtomicU64 = AtomicU64::new(0);
static CUDA_KV_RECLAIMED_PAGES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn allocation_failures_total() -> u64 {
    CUDA_KV_PAGE_ALLOCATION_FAILURES.load(Ordering::Relaxed)
}

pub(crate) fn reclaimed_pages_total() -> u64 {
    CUDA_KV_RECLAIMED_PAGES.load(Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaKvPageStorage {
    F16,
    Q8_0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaKvPageLayout {
    pub(crate) layer_count: usize,
    pub(crate) kv_head_count: usize,
    pub(crate) head_dim: usize,
    pub(crate) storage: CudaKvPageStorage,
}

impl CudaKvPageLayout {
    pub(crate) fn page_bytes(self) -> Result<usize, CudaKvPageError> {
        if self.layer_count == 0 || self.kv_head_count == 0 || self.head_dim == 0 {
            return Err(CudaKvPageError::InvalidLayout);
        }
        let row_bytes = match self.storage {
            CudaKvPageStorage::F16 => self
                .head_dim
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or(CudaKvPageError::LayoutOverflow)?,
            CudaKvPageStorage::Q8_0 => {
                if !self.head_dim.is_multiple_of(32) {
                    return Err(CudaKvPageError::InvalidLayout);
                }
                self.head_dim
                    .checked_div(32)
                    .and_then(|blocks| blocks.checked_mul(34))
                    .ok_or(CudaKvPageError::LayoutOverflow)?
            }
        };
        self.layer_count
            .checked_mul(self.kv_head_count)
            .and_then(|value| value.checked_mul(CUDA_KV_PAGE_TOKENS))
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_mul(row_bytes))
            .ok_or(CudaKvPageError::LayoutOverflow)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CudaKvPageHandle {
    index: u32,
    generation: u64,
}

impl CudaKvPageHandle {
    pub(crate) const fn index(self) -> usize {
        self.index as usize
    }

    pub(crate) const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaKvPageState {
    Vacant,
    Live { references: NonZeroUsize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaKvPageEntry {
    generation: u64,
    state: CudaKvPageState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaKvPagePoolSnapshot {
    pub(crate) layout: CudaKvPageLayout,
    pub(crate) capacity_pages: usize,
    pub(crate) resident_pages: usize,
    pub(crate) allocated_pages: usize,
    pub(crate) free_pages: usize,
    pub(crate) shared_pages: usize,
    pub(crate) high_watermark_pages: usize,
    pub(crate) total_allocations: u64,
    pub(crate) allocation_failures: u64,
    pub(crate) reclaimed_pages: u64,
    pub(crate) page_bytes: usize,
    pub(crate) allocated_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum CudaKvPageError {
    #[error("CUDA KV page layout is invalid")]
    InvalidLayout,
    #[error("CUDA KV page layout byte calculation overflowed")]
    LayoutOverflow,
    #[error("CUDA KV page capacity must be non-zero")]
    ZeroCapacity,
    #[error("CUDA KV page allocation needs {requested} pages but only {available} are free")]
    CapacityExhausted { requested: usize, available: usize },
    #[error("CUDA KV page handle is stale")]
    StaleHandle,
    #[error("CUDA KV page reference count overflowed")]
    ReferenceOverflow,
    #[error("CUDA KV page mutation belongs to another sequence")]
    SequenceMismatch,
    #[error("CUDA KV page table changed after append preparation")]
    ConcurrentMutation,
}

#[derive(Debug)]
pub(crate) struct CudaKvPagePool {
    layout: CudaKvPageLayout,
    page_bytes: usize,
    capacity_pages: usize,
    pages: Vec<CudaKvPageEntry>,
    free: Vec<u32>,
    allocated_pages: usize,
    high_watermark_pages: usize,
    total_allocations: u64,
    allocation_failures: u64,
    reclaimed_pages: u64,
}

impl CudaKvPagePool {
    pub(crate) fn new(
        layout: CudaKvPageLayout,
        capacity_pages: usize,
    ) -> Result<Self, CudaKvPageError> {
        if capacity_pages == 0 {
            return Err(CudaKvPageError::ZeroCapacity);
        }
        let page_bytes = layout.page_bytes()?;
        Ok(Self {
            layout,
            page_bytes,
            capacity_pages,
            pages: Vec::new(),
            free: Vec::new(),
            allocated_pages: 0,
            high_watermark_pages: 0,
            total_allocations: 0,
            allocation_failures: 0,
            reclaimed_pages: 0,
        })
    }

    pub(crate) fn layout(&self) -> CudaKvPageLayout {
        self.layout
    }

    pub(crate) fn reserve(
        &mut self,
        count: usize,
    ) -> Result<Vec<CudaKvPageHandle>, CudaKvPageError> {
        let available = self.capacity_pages.saturating_sub(self.allocated_pages);
        if count > available {
            self.allocation_failures = self.allocation_failures.saturating_add(1);
            CUDA_KV_PAGE_ALLOCATION_FAILURES.fetch_add(1, Ordering::Relaxed);
            return Err(CudaKvPageError::CapacityExhausted {
                requested: count,
                available,
            });
        }
        let mut handles = Vec::with_capacity(count);
        for _ in 0..count {
            handles.push(self.allocate_one());
        }
        Ok(handles)
    }

    fn allocate_one(&mut self) -> CudaKvPageHandle {
        let index = if let Some(index) = self.free.pop() {
            index
        } else {
            let index = self.pages.len() as u32;
            self.pages.push(CudaKvPageEntry {
                generation: 0,
                state: CudaKvPageState::Vacant,
            });
            index
        };
        let entry = &mut self.pages[index as usize];
        debug_assert_eq!(entry.state, CudaKvPageState::Vacant);
        entry.generation = entry.generation.wrapping_add(1).max(1);
        entry.state = CudaKvPageState::Live {
            references: NonZeroUsize::MIN,
        };
        self.allocated_pages += 1;
        self.high_watermark_pages = self.high_watermark_pages.max(self.allocated_pages);
        self.total_allocations = self.total_allocations.saturating_add(1);
        CudaKvPageHandle {
            index,
            generation: entry.generation,
        }
    }

    pub(crate) fn validate(&self, handle: CudaKvPageHandle) -> bool {
        self.pages.get(handle.index()).is_some_and(|entry| {
            entry.generation == handle.generation
                && matches!(entry.state, CudaKvPageState::Live { .. })
        })
    }

    pub(crate) fn references(&self, handle: CudaKvPageHandle) -> Result<usize, CudaKvPageError> {
        let entry = self.live_entry(handle)?;
        let CudaKvPageState::Live { references } = entry.state else {
            unreachable!("live_entry validated the state")
        };
        Ok(references.get())
    }

    fn live_entry(&self, handle: CudaKvPageHandle) -> Result<&CudaKvPageEntry, CudaKvPageError> {
        self.pages
            .get(handle.index())
            .filter(|entry| {
                entry.generation == handle.generation
                    && matches!(entry.state, CudaKvPageState::Live { .. })
            })
            .ok_or(CudaKvPageError::StaleHandle)
    }

    fn live_entry_mut(
        &mut self,
        handle: CudaKvPageHandle,
    ) -> Result<&mut CudaKvPageEntry, CudaKvPageError> {
        self.pages
            .get_mut(handle.index())
            .filter(|entry| {
                entry.generation == handle.generation
                    && matches!(entry.state, CudaKvPageState::Live { .. })
            })
            .ok_or(CudaKvPageError::StaleHandle)
    }

    pub(super) fn retain(&mut self, handle: CudaKvPageHandle) -> Result<(), CudaKvPageError> {
        let entry = self.live_entry_mut(handle)?;
        let CudaKvPageState::Live { references } = &mut entry.state else {
            unreachable!("live_entry_mut validated the state")
        };
        *references = NonZeroUsize::new(
            references
                .get()
                .checked_add(1)
                .ok_or(CudaKvPageError::ReferenceOverflow)?,
        )
        .expect("incrementing a non-zero reference count remains non-zero");
        Ok(())
    }

    fn can_retain(&self, handle: CudaKvPageHandle) -> Result<(), CudaKvPageError> {
        self.references(handle)?
            .checked_add(1)
            .ok_or(CudaKvPageError::ReferenceOverflow)
            .map(|_| ())
    }

    pub(crate) fn release(&mut self, handle: CudaKvPageHandle) -> Result<(), CudaKvPageError> {
        let entry = self.live_entry_mut(handle)?;
        let CudaKvPageState::Live { references } = entry.state else {
            unreachable!("live_entry_mut validated the state")
        };
        if references.get() > 1 {
            entry.state = CudaKvPageState::Live {
                references: NonZeroUsize::new(references.get() - 1)
                    .expect("shared references remain non-zero"),
            };
            return Ok(());
        }
        entry.state = CudaKvPageState::Vacant;
        let index = handle.index;
        self.free.push(index);
        self.allocated_pages -= 1;
        self.reclaimed_pages = self.reclaimed_pages.saturating_add(1);
        CUDA_KV_RECLAIMED_PAGES.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn snapshot(&self) -> CudaKvPagePoolSnapshot {
        let shared_pages = self
            .pages
            .iter()
            .filter(|entry| {
                matches!(
                    entry.state,
                    CudaKvPageState::Live { references } if references.get() > 1
                )
            })
            .count();
        CudaKvPagePoolSnapshot {
            layout: self.layout,
            capacity_pages: self.capacity_pages,
            resident_pages: self.pages.len(),
            allocated_pages: self.allocated_pages,
            free_pages: self.free.len(),
            shared_pages,
            high_watermark_pages: self.high_watermark_pages,
            total_allocations: self.total_allocations,
            allocation_failures: self.allocation_failures,
            reclaimed_pages: self.reclaimed_pages,
            page_bytes: self.page_bytes,
            allocated_bytes: self.allocated_pages.saturating_mul(self.page_bytes),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingPageKind {
    Existing,
    New,
    CopyOnWrite { source: CudaKvPageHandle },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PendingCudaKvAppend {
    sequence_id: u64,
    expected_tokens: usize,
    logical_page: usize,
    token_offset: usize,
    target: CudaKvPageHandle,
    kind: PendingPageKind,
}

impl PendingCudaKvAppend {
    pub(super) const fn target(self) -> CudaKvPageHandle {
        self.target
    }

    pub(super) const fn copy_source(self) -> Option<CudaKvPageHandle> {
        match self.kind {
            PendingPageKind::CopyOnWrite { source } => Some(source),
            PendingPageKind::Existing | PendingPageKind::New => None,
        }
    }

    pub(super) const fn token_offset(self) -> usize {
        self.token_offset
    }
}

#[derive(Debug, Clone)]
pub(super) struct PendingCudaKvBatchAppend {
    sequence_id: u64,
    expected_tokens: usize,
    additions: Vec<PendingCudaKvAppend>,
}

impl PendingCudaKvBatchAppend {
    pub(super) fn allocated_pages(&self) -> impl Iterator<Item = CudaKvPageHandle> + '_ {
        self.additions
            .iter()
            .filter_map(|append| match append.kind {
                PendingPageKind::New | PendingPageKind::CopyOnWrite { .. } => Some(append.target),
                PendingPageKind::Existing => None,
            })
    }
    pub(super) fn copy_on_write_pages(
        &self,
    ) -> impl Iterator<Item = (CudaKvPageHandle, CudaKvPageHandle)> + '_ {
        self.additions
            .iter()
            .filter_map(|append| match append.kind {
                PendingPageKind::CopyOnWrite { source } => Some((source, append.target)),
                PendingPageKind::Existing | PendingPageKind::New => None,
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CudaKvPageTable {
    sequence_id: u64,
    pages: Vec<CudaKvPageHandle>,
    tokens: usize,
}

impl CudaKvPageTable {
    pub(super) fn new(sequence_id: u64) -> Self {
        Self {
            sequence_id,
            pages: Vec::new(),
            tokens: 0,
        }
    }

    pub(super) fn sequence_id(&self) -> u64 {
        self.sequence_id
    }

    pub(super) fn len(&self) -> usize {
        self.tokens
    }

    pub(super) fn pages(&self) -> &[CudaKvPageHandle] {
        &self.pages
    }

    pub(super) fn planned_pages_after_batch(
        &self,
        pending: &PendingCudaKvBatchAppend,
    ) -> Result<Vec<CudaKvPageHandle>, CudaKvPageError> {
        if pending.sequence_id != self.sequence_id || pending.expected_tokens != self.tokens {
            return Err(CudaKvPageError::ConcurrentMutation);
        }
        let mut pages = self.pages.clone();
        for addition in &pending.additions {
            match addition.kind {
                PendingPageKind::Existing => {
                    if pages.get(addition.logical_page) != Some(&addition.target) {
                        return Err(CudaKvPageError::ConcurrentMutation);
                    }
                }
                PendingPageKind::New => {
                    if addition.logical_page != pages.len() {
                        return Err(CudaKvPageError::ConcurrentMutation);
                    }
                    pages.push(addition.target);
                }
                PendingPageKind::CopyOnWrite { source } => {
                    let page = pages
                        .get_mut(addition.logical_page)
                        .filter(|page| **page == source)
                        .ok_or(CudaKvPageError::ConcurrentMutation)?;
                    *page = addition.target;
                }
            }
        }
        Ok(pages)
    }

    pub(super) fn logical_to_physical(&self, position: usize) -> Option<(CudaKvPageHandle, usize)> {
        if position >= self.tokens {
            return None;
        }
        self.pages
            .get(position / CUDA_KV_PAGE_TOKENS)
            .copied()
            .map(|page| (page, position % CUDA_KV_PAGE_TOKENS))
    }

    pub(super) fn prepare_append(
        &self,
        pool: &mut CudaKvPagePool,
    ) -> Result<PendingCudaKvAppend, CudaKvPageError> {
        let logical_page = self.tokens / CUDA_KV_PAGE_TOKENS;
        let token_offset = self.tokens % CUDA_KV_PAGE_TOKENS;
        let (target, kind) = if logical_page == self.pages.len() {
            let target = pool.reserve(1)?.pop().expect("one page was reserved");
            (target, PendingPageKind::New)
        } else {
            let source = self.pages[logical_page];
            let references = pool.references(source)?;
            if references == 1 {
                (source, PendingPageKind::Existing)
            } else {
                let target = pool.reserve(1)?.pop().expect("one page was reserved");
                (target, PendingPageKind::CopyOnWrite { source })
            }
        };
        Ok(PendingCudaKvAppend {
            sequence_id: self.sequence_id,
            expected_tokens: self.tokens,
            logical_page,
            token_offset,
            target,
            kind,
        })
    }

    pub(super) fn prepare_append_batch(
        &self,
        pool: &mut CudaKvPagePool,
        count: usize,
    ) -> Result<PendingCudaKvBatchAppend, CudaKvPageError> {
        let mut simulation = self.clone();
        let mut additions = Vec::with_capacity(count);
        for _ in 0..count {
            let pending = match simulation.prepare_append(pool) {
                Ok(pending) => pending,
                Err(error) => {
                    for reserved in additions.into_iter().rev() {
                        Self::abort_append(pool, reserved)?;
                    }
                    return Err(error);
                }
            };
            match pending.kind {
                PendingPageKind::New => simulation.pages.push(pending.target),
                PendingPageKind::CopyOnWrite { source } => {
                    let page = simulation
                        .pages
                        .get_mut(pending.logical_page)
                        .filter(|page| **page == source)
                        .ok_or(CudaKvPageError::ConcurrentMutation)?;
                    *page = pending.target;
                }
                PendingPageKind::Existing => {}
            }
            simulation.tokens += 1;
            additions.push(pending);
        }
        Ok(PendingCudaKvBatchAppend {
            sequence_id: self.sequence_id,
            expected_tokens: self.tokens,
            additions,
        })
    }

    pub(super) fn commit_append(
        &mut self,
        pool: &mut CudaKvPagePool,
        pending: PendingCudaKvAppend,
    ) -> Result<(CudaKvPageHandle, usize), CudaKvPageError> {
        self.validate_append(pool, pending)?;
        match pending.kind {
            PendingPageKind::Existing => {}
            PendingPageKind::New => self.pages.push(pending.target),
            PendingPageKind::CopyOnWrite { source } => {
                pool.release(source)?;
                self.pages[pending.logical_page] = pending.target;
            }
        }
        self.tokens += 1;
        Ok((pending.target, pending.token_offset))
    }

    fn validate_append(
        &self,
        pool: &CudaKvPagePool,
        pending: PendingCudaKvAppend,
    ) -> Result<(), CudaKvPageError> {
        if pending.sequence_id != self.sequence_id {
            return Err(CudaKvPageError::SequenceMismatch);
        }
        if pending.expected_tokens != self.tokens
            || pending.logical_page != self.tokens / CUDA_KV_PAGE_TOKENS
            || pending.token_offset != self.tokens % CUDA_KV_PAGE_TOKENS
        {
            return Err(CudaKvPageError::ConcurrentMutation);
        }
        if !pool.validate(pending.target) {
            return Err(CudaKvPageError::StaleHandle);
        }
        match pending.kind {
            PendingPageKind::Existing => {
                if self.pages.get(pending.logical_page) != Some(&pending.target) {
                    return Err(CudaKvPageError::ConcurrentMutation);
                }
            }
            PendingPageKind::New => {
                if pending.logical_page != self.pages.len() {
                    return Err(CudaKvPageError::ConcurrentMutation);
                }
            }
            PendingPageKind::CopyOnWrite { source } => {
                if self.pages.get(pending.logical_page) != Some(&source)
                    || pool.references(source)? < 2
                {
                    return Err(CudaKvPageError::ConcurrentMutation);
                }
            }
        }
        Ok(())
    }

    pub(super) fn commit_append_batch(
        &mut self,
        pool: &mut CudaKvPagePool,
        pending: PendingCudaKvBatchAppend,
    ) -> Result<(), CudaKvPageError> {
        if pending.sequence_id != self.sequence_id || pending.expected_tokens != self.tokens {
            return Err(CudaKvPageError::ConcurrentMutation);
        }
        let mut simulation = self.clone();
        for addition in &pending.additions {
            simulation.validate_append(pool, *addition)?;
            match addition.kind {
                PendingPageKind::Existing => {}
                PendingPageKind::New => simulation.pages.push(addition.target),
                PendingPageKind::CopyOnWrite { .. } => {
                    simulation.pages[addition.logical_page] = addition.target;
                }
            }
            simulation.tokens += 1;
        }
        for addition in pending.additions {
            self.commit_append(pool, addition)?;
        }
        Ok(())
    }

    #[allow(dead_code)] // Used by the paged serving transaction once kernel routing is enabled.
    pub(super) fn abort_append_batch(
        pool: &mut CudaKvPagePool,
        pending: PendingCudaKvBatchAppend,
    ) -> Result<(), CudaKvPageError> {
        for addition in pending.additions.into_iter().rev() {
            Self::abort_append(pool, addition)?;
        }
        Ok(())
    }

    pub(super) fn abort_append(
        pool: &mut CudaKvPagePool,
        pending: PendingCudaKvAppend,
    ) -> Result<(), CudaKvPageError> {
        if matches!(
            pending.kind,
            PendingPageKind::New | PendingPageKind::CopyOnWrite { .. }
        ) {
            pool.release(pending.target)?;
        }
        Ok(())
    }

    pub(super) fn fork(
        &self,
        pool: &mut CudaKvPagePool,
        child_sequence_id: u64,
    ) -> Result<Self, CudaKvPageError> {
        for page in &self.pages {
            pool.can_retain(*page)?;
        }
        for page in &self.pages {
            pool.retain(*page)?;
        }
        Ok(Self {
            sequence_id: child_sequence_id,
            pages: self.pages.clone(),
            tokens: self.tokens,
        })
    }

    pub(super) fn release_all(&mut self, pool: &mut CudaKvPagePool) -> Result<(), CudaKvPageError> {
        for page in &self.pages {
            pool.live_entry(*page)?;
        }
        for page in self.pages.drain(..) {
            pool.release(page)?;
        }
        self.tokens = 0;
        Ok(())
    }

    pub(super) fn truncate(
        &mut self,
        pool: &mut CudaKvPagePool,
        tokens: usize,
    ) -> Result<(), CudaKvPageError> {
        if tokens >= self.tokens {
            return Ok(());
        }
        let keep_pages = tokens.div_ceil(CUDA_KV_PAGE_TOKENS);
        for page in &self.pages[keep_pages..] {
            pool.live_entry(*page)?;
        }
        let removed = self.pages.split_off(keep_pages);
        for page in removed {
            pool.release(page)?;
        }
        self.tokens = tokens;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> CudaKvPageLayout {
        CudaKvPageLayout {
            layer_count: 16,
            kv_head_count: 8,
            head_dim: 64,
            storage: CudaKvPageStorage::F16,
        }
    }

    fn append(table: &mut CudaKvPageTable, pool: &mut CudaKvPagePool) {
        let pending = table.prepare_append(pool).unwrap();
        table.commit_append(pool, pending).unwrap();
    }

    #[test]
    fn page_layout_bytes_are_exact_and_checked() {
        assert_eq!(layout().page_bytes().unwrap(), 524_288);
        assert_eq!(
            CudaKvPageLayout {
                storage: CudaKvPageStorage::Q8_0,
                ..layout()
            }
            .page_bytes()
            .unwrap(),
            278_528
        );
        assert_eq!(
            CudaKvPageLayout {
                head_dim: 63,
                storage: CudaKvPageStorage::Q8_0,
                ..layout()
            }
            .page_bytes(),
            Err(CudaKvPageError::InvalidLayout)
        );
        let pool = CudaKvPagePool::new(layout(), 2).unwrap();
        assert_eq!(pool.layout(), layout());
        assert_eq!(pool.snapshot().layout, layout());
    }

    #[test]
    fn allocation_release_reuse_is_generation_safe_and_never_aliases() {
        let mut pool = CudaKvPagePool::new(layout(), 2).unwrap();
        let first = pool.reserve(1).unwrap()[0];
        pool.release(first).unwrap();
        assert_eq!(pool.release(first), Err(CudaKvPageError::StaleHandle));

        let reused = pool.reserve(1).unwrap()[0];
        let sibling = pool.reserve(1).unwrap()[0];
        assert_eq!(reused.index(), first.index());
        assert_ne!(reused.generation(), first.generation());
        assert_ne!(reused, sibling);
        assert!(!pool.validate(first));
    }

    #[test]
    fn failed_multi_page_reservation_is_atomic() {
        let mut pool = CudaKvPagePool::new(layout(), 3).unwrap();
        pool.reserve(2).unwrap();
        let before = pool.snapshot();
        assert_eq!(
            pool.reserve(2),
            Err(CudaKvPageError::CapacityExhausted {
                requested: 2,
                available: 1,
            })
        );
        let after = pool.snapshot();
        assert_eq!(after.allocated_pages, before.allocated_pages);
        assert_eq!(after.allocated_bytes, before.allocated_bytes);
        assert_eq!(after.resident_pages, before.resident_pages);
        assert_eq!(after.free_pages, before.free_pages);
        assert_eq!(after.shared_pages, before.shared_pages);
        assert_eq!(after.high_watermark_pages, before.high_watermark_pages);
        assert_eq!(after.total_allocations, before.total_allocations);
        assert_eq!(after.reclaimed_pages, before.reclaimed_pages);
        assert_eq!(after.allocation_failures, before.allocation_failures + 1);
    }

    #[test]
    fn forked_partial_page_uses_transactional_copy_on_write() {
        let mut pool = CudaKvPagePool::new(layout(), 4).unwrap();
        let mut parent = CudaKvPageTable::new(10);
        for _ in 0..5 {
            append(&mut parent, &mut pool);
        }
        let mut child = parent.fork(&mut pool, 20).unwrap();
        let shared = parent.pages()[0];
        assert_eq!(pool.references(shared), Ok(2));

        let pending = child.prepare_append(&mut pool).unwrap();
        assert_eq!(pending.copy_source(), Some(shared));
        assert_ne!(pending.target(), shared);
        assert_eq!(pending.token_offset(), 5);
        assert_eq!(
            parent.pages()[0],
            shared,
            "prepare cannot mutate the parent"
        );
        assert_eq!(child.pages()[0], shared, "prepare cannot mutate the child");
        assert_eq!(pool.references(shared), Ok(2));

        let private = pending.target();
        child.commit_append(&mut pool, pending).unwrap();
        assert_eq!(parent.pages()[0], shared);
        assert_eq!(child.pages()[0], private);
        assert_eq!(pool.references(shared), Ok(1));
        assert_eq!(pool.references(private), Ok(1));
    }

    #[test]
    fn aborted_copy_on_write_leaves_both_tables_and_source_unchanged() {
        let mut pool = CudaKvPagePool::new(layout(), 3).unwrap();
        let mut parent = CudaKvPageTable::new(1);
        append(&mut parent, &mut pool);
        let child = parent.fork(&mut pool, 2).unwrap();
        let source = parent.pages()[0];
        let before_parent = parent.clone();
        let before_child = child.clone();
        let pending = child.prepare_append(&mut pool).unwrap();
        CudaKvPageTable::abort_append(&mut pool, pending).unwrap();
        assert_eq!(parent, before_parent);
        assert_eq!(child, before_child);
        assert_eq!(pool.references(source), Ok(2));
        assert_eq!(pool.snapshot().allocated_pages, 1);
    }

    #[test]
    fn active_pages_are_pinned_and_capacity_never_evicts_them() {
        let mut pool = CudaKvPagePool::new(layout(), 1).unwrap();
        let live = pool.reserve(1).unwrap()[0];
        assert_eq!(
            pool.reserve(1),
            Err(CudaKvPageError::CapacityExhausted {
                requested: 1,
                available: 0,
            })
        );
        assert!(pool.validate(live));
        assert_eq!(pool.references(live), Ok(1));
    }

    #[test]
    fn cancelled_sequence_reclaims_pages_for_the_next_round() {
        let mut pool = CudaKvPagePool::new(layout(), 2).unwrap();
        let mut cancelled = CudaKvPageTable::new(7);
        for _ in 0..17 {
            append(&mut cancelled, &mut pool);
        }
        let stale = cancelled.pages().to_vec();
        assert_eq!(pool.snapshot().allocated_pages, 2);
        cancelled.release_all(&mut pool).unwrap();
        assert_eq!(pool.snapshot().allocated_pages, 0);

        let mut next = CudaKvPageTable::new(8);
        for _ in 0..17 {
            append(&mut next, &mut pool);
        }
        assert_eq!(pool.snapshot().allocated_pages, 2);
        assert!(stale.iter().all(|page| !pool.validate(*page)));
        assert!(next.pages().iter().all(|page| pool.validate(*page)));
    }

    #[test]
    fn idle_accounting_reconciles_and_missing_release_is_detectable() {
        let mut pool = CudaKvPagePool::new(layout(), 4).unwrap();
        let mut table = CudaKvPageTable::new(99);
        for _ in 0..33 {
            append(&mut table, &mut pool);
        }
        let leaked = pool.snapshot();
        assert_eq!(leaked.allocated_pages, 3);
        assert_ne!(
            leaked.allocated_bytes, 0,
            "missing release must fail idle accounting"
        );

        table.release_all(&mut pool).unwrap();
        let idle = pool.snapshot();
        assert_eq!(idle.allocated_pages, 0);
        assert_eq!(idle.allocated_bytes, 0);
        assert_eq!(idle.reclaimed_pages, 3);
        assert_eq!(idle.high_watermark_pages, 3);
        assert_eq!(idle.total_allocations, 3);
        assert_eq!(idle.free_pages, 3);
        assert_eq!(idle.resident_pages, 3);
    }

    #[test]
    fn sequence_table_maps_positions_without_cross_sequence_aliasing() {
        let mut pool = CudaKvPagePool::new(layout(), 4).unwrap();
        let mut first = CudaKvPageTable::new(1);
        let mut second = CudaKvPageTable::new(2);
        for _ in 0..17 {
            append(&mut first, &mut pool);
            append(&mut second, &mut pool);
        }
        assert_eq!(first.sequence_id(), 1);
        assert_eq!(second.sequence_id(), 2);
        assert_eq!(first.len(), 17);
        assert_eq!(second.len(), 17);
        assert_ne!(first.pages(), second.pages());
        assert_eq!(first.logical_to_physical(16).unwrap().1, 0);
        assert_eq!(second.logical_to_physical(16).unwrap().1, 0);
    }

    #[test]
    fn batch_append_failure_rolls_back_every_reserved_page() {
        let mut pool = CudaKvPagePool::new(layout(), 2).unwrap();
        let table = CudaKvPageTable::new(1);
        let before = pool.snapshot();
        assert!(matches!(
            table.prepare_append_batch(&mut pool, 33),
            Err(CudaKvPageError::CapacityExhausted { .. })
        ));
        assert_eq!(pool.snapshot().allocated_pages, before.allocated_pages);
        assert_eq!(
            pool.snapshot().allocation_failures,
            before.allocation_failures + 1
        );
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn batch_append_and_truncate_reconcile_pages() {
        let mut pool = CudaKvPagePool::new(layout(), 4).unwrap();
        let mut table = CudaKvPageTable::new(1);
        let pending = table.prepare_append_batch(&mut pool, 33).unwrap();
        assert_eq!(pending.allocated_pages().count(), 3);
        table.commit_append_batch(&mut pool, pending).unwrap();
        assert_eq!(table.len(), 33);
        assert_eq!(table.pages().len(), 3);
        table.truncate(&mut pool, 16).unwrap();
        assert_eq!(table.len(), 16);
        assert_eq!(table.pages().len(), 1);
        assert_eq!(pool.snapshot().allocated_pages, 1);
    }

    #[test]
    fn planned_batch_pages_include_new_pages_without_committing_the_table() {
        let mut pool = CudaKvPagePool::new(layout(), 4).unwrap();
        let mut table = CudaKvPageTable::new(1);
        let pending = table.prepare_append_batch(&mut pool, 17).unwrap();
        let planned = table.planned_pages_after_batch(&pending).unwrap();
        assert_eq!(planned.len(), 2);
        assert_eq!(table.len(), 0);
        assert!(table.pages().is_empty());
        assert_eq!(pool.snapshot().allocated_pages, 2);
        table.commit_append_batch(&mut pool, pending).unwrap();
        assert_eq!(table.pages(), planned);
        assert_eq!(table.len(), 17);
    }

    #[test]
    fn corrupted_later_batch_addition_cannot_partially_commit_the_table() {
        let mut pool = CudaKvPagePool::new(layout(), 2).unwrap();
        let mut table = CudaKvPageTable::new(1);
        let mut pending = table.prepare_append_batch(&mut pool, 17).unwrap();
        pending.additions[1].expected_tokens += 1;
        let table_before = table.clone();
        let pool_before = pool.snapshot();

        assert_eq!(
            table.commit_append_batch(&mut pool, pending.clone()),
            Err(CudaKvPageError::ConcurrentMutation)
        );
        assert_eq!(table, table_before);
        assert_eq!(pool.snapshot().allocated_pages, pool_before.allocated_pages);
        assert_eq!(pool.snapshot().allocated_bytes, pool_before.allocated_bytes);

        CudaKvPageTable::abort_append_batch(&mut pool, pending).unwrap();
        assert_eq!(pool.snapshot().allocated_pages, 0);
        assert_eq!(pool.snapshot().allocated_bytes, 0);
    }
}
