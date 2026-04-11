// Qualia shared memory: POSIX shm for macOS with double-buffered layer slots and ring-buffer ledger.

pub use qualia_types::*;

use std::ffi::CString;
use std::sync::atomic::Ordering;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

pub enum ShmError {
    OsError(i32),
    BadMagic,
    SizeMismatch,
}

impl std::fmt::Debug for ShmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShmError::OsError(e) => write!(f, "ShmError::OsError({})", e),
            ShmError::BadMagic => write!(f, "ShmError::BadMagic"),
            ShmError::SizeMismatch => write!(f, "ShmError::SizeMismatch"),
        }
    }
}

impl std::fmt::Display for ShmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShmError::OsError(e) => write!(f, "OS error {}", e),
            ShmError::BadMagic => write!(f, "bad magic number in shared memory header"),
            ShmError::SizeMismatch => write!(f, "shared memory size mismatch"),
        }
    }
}

impl std::error::Error for ShmError {}

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

/// Total shared memory region: 64 MiB.
pub const SHM_SIZE: usize = 64 * 1024 * 1024;

pub const HEADER_SIZE: usize = std::mem::size_of::<ShmHeader>();
pub const LAYER_SLOT_SIZE: usize = std::mem::size_of::<LayerSlot>();

/// Header is page-aligned at offset 0; layer slots start at offset 4096.
pub const LAYER_SLOTS_OFFSET: usize = 4096;

/// Ledger starts after header page + all layer slots.
pub const LEDGER_OFFSET: usize = LAYER_SLOTS_OFFSET + NUM_LAYERS * LAYER_SLOT_SIZE;

/// Ledger region: 16 MiB.
pub const LEDGER_SIZE: usize = 16 * 1024 * 1024;

/// Maximum number of ledger entries (ring buffer).
pub const MAX_LEDGER_ENTRIES: usize = LEDGER_SIZE / std::mem::size_of::<LedgerEntry>();

/// WorldModel lives after the ledger.
pub const WORLD_MODEL_OFFSET: usize = LEDGER_OFFSET + LEDGER_SIZE;

/// ThoughtBuffer lives after WorldModel.
pub const THOUGHT_BUFFER_OFFSET: usize = WORLD_MODEL_OFFSET + std::mem::size_of::<WorldModel>();

/// LoreBuffer lives after ThoughtBuffer.
pub const LORE_BUFFER_OFFSET: usize = THOUGHT_BUFFER_OFFSET + std::mem::size_of::<ThoughtBuffer>();

/// ActionHistory lives after LoreBuffer (Phase 3.2).
pub const ACTION_HISTORY_OFFSET: usize = LORE_BUFFER_OFFSET + std::mem::size_of::<LoreBuffer>();

// ── Thought Theater layout ──────────────────────────────────────────
pub const THEATER_OFFSET: usize = ACTION_HISTORY_OFFSET + std::mem::size_of::<ActionHistory>();
pub const BIG_GRAPH_OFFSET: usize = THEATER_OFFSET + std::mem::size_of::<TheaterHeader>();
pub const SMALL_GRAPH_OFFSET: usize = BIG_GRAPH_OFFSET + std::mem::size_of::<BigGraph>();
pub const OP_SLICE_OFFSET: usize = SMALL_GRAPH_OFFSET + std::mem::size_of::<SmallGraph>();
pub const PRESSURE_OFFSET: usize = OP_SLICE_OFFSET + std::mem::size_of::<OperationalSlice>();
pub const TRIM_MASK_OFFSET: usize = PRESSURE_OFFSET + std::mem::size_of::<PressureBuffer>();

// ---------------------------------------------------------------------------
// ShmRegion
// ---------------------------------------------------------------------------

pub struct ShmRegion {
    ptr: *mut u8,
    len: usize,
    name: String,
    owner: bool,
}

// SAFETY: All mutable shared state is accessed exclusively through atomic
// operations (AtomicUsize, AtomicBool, AtomicU64) defined in the repr(C)
// types. The raw pointer is to a memory-mapped region that outlives all
// references derived from it.
unsafe impl Send for ShmRegion {}
unsafe impl Sync for ShmRegion {}

impl ShmRegion {
    /// Create a new shared memory region. Only the supervisor should call this.
    pub fn create(name: &str) -> Result<Self, ShmError> {
        let c_name = CString::new(name).map_err(|_| ShmError::OsError(libc::EINVAL))?;

        // SAFETY: shm_open with O_CREAT | O_RDWR | O_EXCL creates a new POSIX
        // shared memory object. The c_name is a valid null-terminated C string.
        let fd = unsafe {
            libc::shm_open(
                c_name.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_EXCL,
                0o600,
            )
        };
        if fd < 0 {
            return Err(ShmError::OsError(errno()));
        }

        // SAFETY: ftruncate sets the shared memory object size. fd is valid.
        if unsafe { libc::ftruncate(fd, SHM_SIZE as libc::off_t) } != 0 {
            let e = errno();
            unsafe {
                libc::close(fd);
                libc::shm_unlink(c_name.as_ptr());
            }
            return Err(ShmError::OsError(e));
        }

        // SAFETY: mmap maps the shared memory into our address space. We pass
        // valid fd, size, and flags. MAP_SHARED allows cross-process sharing.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                SHM_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };

        // SAFETY: close the fd; the mapping keeps the region alive.
        unsafe { libc::close(fd) };

        if ptr == libc::MAP_FAILED {
            let e = errno();
            unsafe { libc::shm_unlink(c_name.as_ptr()) };
            return Err(ShmError::OsError(e));
        }

        let ptr = ptr as *mut u8;

        // SAFETY: We just mmap'd SHM_SIZE bytes at ptr; zeroing is safe.
        unsafe { std::ptr::write_bytes(ptr, 0, SHM_SIZE) };

        // SAFETY: ptr is aligned to at least page boundary (4096), which
        // exceeds ShmHeader alignment. We write the header fields.
        let header = unsafe { &mut *(ptr as *mut ShmHeader) };
        header.magic = SHM_MAGIC;
        header.version = 1;
        header.num_layers = NUM_LAYERS as u32;
        header.layer_slot_size = LAYER_SLOT_SIZE as u64;
        header.ledger_offset = LEDGER_OFFSET as u64;
        header.ledger_capacity = MAX_LEDGER_ENTRIES as u64;
        header.ledger_write_seq.store(0, Ordering::Release);

        Ok(Self {
            ptr,
            len: SHM_SIZE,
            name: name.to_string(),
            owner: true,
        })
    }

    /// Open an existing shared memory region. Runners call this.
    pub fn open(name: &str) -> Result<Self, ShmError> {
        let c_name = CString::new(name).map_err(|_| ShmError::OsError(libc::EINVAL))?;

        // SAFETY: shm_open without O_CREAT opens an existing region.
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0) };
        if fd < 0 {
            return Err(ShmError::OsError(errno()));
        }

        // SAFETY: mmap the existing region.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                SHM_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };

        // SAFETY: close fd; mapping keeps the region alive.
        unsafe { libc::close(fd) };

        if ptr == libc::MAP_FAILED {
            return Err(ShmError::OsError(errno()));
        }

        let ptr = ptr as *mut u8;

        // SAFETY: ptr points to a valid ShmHeader written by the creator.
        let header = unsafe { &*(ptr as *const ShmHeader) };
        if header.magic != SHM_MAGIC {
            unsafe { libc::munmap(ptr as *mut libc::c_void, SHM_SIZE) };
            return Err(ShmError::BadMagic);
        }

        Ok(Self {
            ptr,
            len: SHM_SIZE,
            name: name.to_string(),
            owner: false,
        })
    }

    /// Reference to the shared memory header.
    pub fn header(&self) -> &ShmHeader {
        // SAFETY: ptr is page-aligned and points to a valid ShmHeader.
        unsafe { &*(self.ptr as *const ShmHeader) }
    }

    /// Reference to the layer slot for `layer` (0..NUM_LAYERS).
    pub fn layer_slot(&self, layer: usize) -> &LayerSlot {
        assert!(layer < NUM_LAYERS, "layer index {} out of range", layer);
        // SAFETY: layer slots begin at LAYER_SLOTS_OFFSET. Each LayerSlot is
        // repr(C) and the region was zeroed on creation.
        unsafe {
            let slot_ptr = self.ptr.add(LAYER_SLOTS_OFFSET + layer * LAYER_SLOT_SIZE);
            &*(slot_ptr as *const LayerSlot)
        }
    }

    /// Reference to a ledger entry by absolute index within the ring buffer.
    pub fn ledger_entry(&self, index: usize) -> &LedgerEntry {
        assert!(index < MAX_LEDGER_ENTRIES, "ledger index out of range");
        // SAFETY: index is bounds-checked; entry layout is repr(C).
        unsafe {
            let entry_ptr = self
                .ptr
                .add(LEDGER_OFFSET + index * std::mem::size_of::<LedgerEntry>());
            &*(entry_ptr as *const LedgerEntry)
        }
    }

    /// Append a ledger entry to the ring buffer (lock-free).
    pub fn append_ledger(&self, entry: &LedgerEntry) {
        let seq = self
            .header()
            .ledger_write_seq
            .fetch_add(1, Ordering::AcqRel);
        let slot_index = (seq as usize) % MAX_LEDGER_ENTRIES;
        // SAFETY: slot_index is in bounds. We write a Copy type.
        unsafe {
            let dst = self
                .ptr
                .add(LEDGER_OFFSET + slot_index * std::mem::size_of::<LedgerEntry>())
                as *mut LedgerEntry;
            std::ptr::write(dst, *entry);
        }
    }

    /// Current ledger write sequence number.
    pub fn ledger_seq(&self) -> u64 {
        self.header().ledger_write_seq.load(Ordering::Acquire)
    }

    /// Reference to the WorldModel in shared memory.
    pub fn world_model(&self) -> &WorldModel {
        unsafe {
            let ptr = self.ptr.add(WORLD_MODEL_OFFSET);
            &*(ptr as *const WorldModel)
        }
    }

    /// Mutable reference to the WorldModel. Only the vision runner should call this.
    pub fn world_model_mut(&self) -> &mut WorldModel {
        unsafe {
            let ptr = self.ptr.add(WORLD_MODEL_OFFSET);
            &mut *(ptr as *mut WorldModel)
        }
    }

    /// Reference to the ThoughtBuffer.
    pub fn thought_buffer(&self) -> &ThoughtBuffer {
        unsafe {
            let ptr = self.ptr.add(THOUGHT_BUFFER_OFFSET);
            &*(ptr as *const ThoughtBuffer)
        }
    }

    /// Append a thought to the ring buffer (lock-free).
    pub fn emit_thought(&self, layer: u8, kind: u8, vfe: f32, text: &str) {
        let tb = unsafe {
            let ptr = self.ptr.add(THOUGHT_BUFFER_OFFSET);
            &mut *(ptr as *mut ThoughtBuffer)
        };
        let seq = tb.write_seq.fetch_add(1, Ordering::AcqRel);
        let idx = (seq as usize) % MAX_THOUGHTS;
        let entry = &mut tb.entries[idx];
        entry.seq = seq;
        entry.layer = layer;
        entry.kind = kind;
        entry.vfe = vfe;
        entry.timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        entry.text = [0u8; MAX_THOUGHT_LEN];
        let bytes = text.as_bytes();
        let len = bytes.len().min(MAX_THOUGHT_LEN - 1);
        entry.text[..len].copy_from_slice(&bytes[..len]);
    }

    /// Reference to the LoreBuffer.
    pub fn lore_buffer(&self) -> &LoreBuffer {
        unsafe {
            let ptr = self.ptr.add(LORE_BUFFER_OFFSET);
            &*(ptr as *const LoreBuffer)
        }
    }

    /// Append a lore entry to the ring buffer (lock-free).
    pub fn emit_lore(&self, question: &str, answer: &str, layer: u8, reason: u8, embedding_delta: f32, effectiveness: f32) {
        let lb = unsafe {
            let ptr = self.ptr.add(LORE_BUFFER_OFFSET);
            &mut *(ptr as *mut LoreBuffer)
        };
        let seq = lb.write_seq.fetch_add(1, Ordering::AcqRel);
        let idx = (seq as usize) % MAX_LORE_ENTRIES;
        let entry = &mut lb.entries[idx];
        entry.seq = seq;
        entry.layer = layer;
        entry.reason = reason;
        entry.embedding_delta = embedding_delta;
        entry.effectiveness = effectiveness;
        entry.timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        entry.question = [0u8; MAX_LORE_QUESTION];
        let q_bytes = question.as_bytes();
        let q_len = q_bytes.len().min(MAX_LORE_QUESTION - 1);
        entry.question[..q_len].copy_from_slice(&q_bytes[..q_len]);

        entry.answer = [0u8; MAX_LORE_TEXT];
        let a_bytes = answer.as_bytes();
        let a_len = a_bytes.len().min(MAX_LORE_TEXT - 1);
        entry.answer[..a_len].copy_from_slice(&a_bytes[..a_len]);
    }

    /// Reference to the ActionHistory (Phase 3.2).
    pub fn action_history(&self) -> &ActionHistory {
        unsafe {
            let ptr = self.ptr.add(ACTION_HISTORY_OFFSET);
            &*(ptr as *const ActionHistory)
        }
    }

    /// Mutable reference to the ActionHistory. Only the explorer should call this.
    pub fn action_history_mut(&self) -> &mut ActionHistory {
        unsafe {
            let ptr = self.ptr.add(ACTION_HISTORY_OFFSET);
            &mut *(ptr as *mut ActionHistory)
        }
    }

    // ── Thought Theater accessors ────────────────────────────────────

    pub fn theater_header(&self) -> &TheaterHeader {
        unsafe {
            let ptr = self.ptr.add(THEATER_OFFSET);
            &*(ptr as *const TheaterHeader)
        }
    }

    pub fn big_graph(&self) -> &BigGraph {
        unsafe {
            let ptr = self.ptr.add(BIG_GRAPH_OFFSET);
            &*(ptr as *const BigGraph)
        }
    }

    pub fn big_graph_mut(&self) -> &mut BigGraph {
        unsafe {
            let ptr = self.ptr.add(BIG_GRAPH_OFFSET);
            &mut *(ptr as *mut BigGraph)
        }
    }

    pub fn small_graph(&self) -> &SmallGraph {
        unsafe {
            let ptr = self.ptr.add(SMALL_GRAPH_OFFSET);
            &*(ptr as *const SmallGraph)
        }
    }

    pub fn small_graph_mut(&self) -> &mut SmallGraph {
        unsafe {
            let ptr = self.ptr.add(SMALL_GRAPH_OFFSET);
            &mut *(ptr as *mut SmallGraph)
        }
    }

    pub fn operational_slice(&self) -> &OperationalSlice {
        unsafe {
            let ptr = self.ptr.add(OP_SLICE_OFFSET);
            &*(ptr as *const OperationalSlice)
        }
    }

    pub fn operational_slice_mut(&self) -> &mut OperationalSlice {
        unsafe {
            let ptr = self.ptr.add(OP_SLICE_OFFSET);
            &mut *(ptr as *mut OperationalSlice)
        }
    }

    pub fn pressure_buffer(&self) -> &PressureBuffer {
        unsafe {
            let ptr = self.ptr.add(PRESSURE_OFFSET);
            &*(ptr as *const PressureBuffer)
        }
    }

    pub fn pressure_buffer_mut(&self) -> &mut PressureBuffer {
        unsafe {
            let ptr = self.ptr.add(PRESSURE_OFFSET);
            &mut *(ptr as *mut PressureBuffer)
        }
    }

    pub fn trim_mask(&self) -> &TrimMask {
        unsafe {
            let ptr = self.ptr.add(TRIM_MASK_OFFSET);
            &*(ptr as *const TrimMask)
        }
    }

    pub fn trim_mask_mut(&self) -> &mut TrimMask {
        unsafe {
            let ptr = self.ptr.add(TRIM_MASK_OFFSET);
            &mut *(ptr as *mut TrimMask)
        }
    }

    /// Append an action entry to the ring buffer (lock-free).
    pub fn emit_action(&self, entry: &ActionEntry) {
        let ah = self.action_history_mut();
        let seq = ah.write_seq.fetch_add(1, Ordering::AcqRel);
        let idx = (seq as usize) % MAX_ACTION_ENTRIES;
        ah.entries[idx] = *entry;
    }

    /// Raw pointer to the mapped region.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Size of the mapped region in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the region is empty (always false for a valid region).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for ShmRegion {
    fn drop(&mut self) {
        // SAFETY: ptr and len were set from a successful mmap call.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
        if self.owner {
            if let Ok(c_name) = CString::new(self.name.as_str()) {
                // SAFETY: c_name is a valid null-terminated string.
                unsafe {
                    libc::shm_unlink(c_name.as_ptr());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
}

// ---------------------------------------------------------------------------
// LayerWriter — write access to a layer's double buffer
// ---------------------------------------------------------------------------

pub struct LayerWriter<'a> {
    slot: &'a LayerSlot,
}

impl<'a> LayerWriter<'a> {
    pub fn new(slot: &'a LayerSlot) -> Self {
        Self { slot }
    }

    /// Returns a mutable reference to the **back buffer** (the one NOT being
    /// read by consumers). Writers fill this buffer, then call [`publish`].
    pub fn back_buffer(&self) -> &mut BeliefSlot {
        let write_idx = self.slot.write_idx.load(Ordering::Acquire) & 1;
        let back = 1 - write_idx;
        // SAFETY: Only one writer exists per layer. The back buffer is not
        // being read by any reader (readers read write_idx, we write 1 - write_idx).
        // We use ptr arithmetic from the slot pointer to avoid casting &T to &mut T.
        unsafe {
            let slot_ptr = self.slot as *const LayerSlot as *mut LayerSlot;
            &mut (*slot_ptr).buffers[back]
        }
    }

    /// Atomically swap: make the back buffer the new front buffer so readers
    /// see the updated data.
    pub fn publish(&self) {
        let old = self.slot.write_idx.load(Ordering::Acquire) & 1;
        self.slot.write_idx.store(1 - old, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// LayerReader — read access to a layer's front buffer
// ---------------------------------------------------------------------------

pub struct LayerReader<'a> {
    slot: &'a LayerSlot,
}

impl<'a> LayerReader<'a> {
    pub fn new(slot: &'a LayerSlot) -> Self {
        Self { slot }
    }

    /// Returns a reference to the **front buffer** — the most recently
    /// published belief state.
    pub fn read(&self) -> &BeliefSlot {
        let write_idx = self.slot.write_idx.load(Ordering::Acquire) & 1;
        &self.slot.buffers[write_idx]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn layout_constants_are_sane() {
        assert!(LEDGER_OFFSET < SHM_SIZE);
        assert!(LEDGER_OFFSET + LEDGER_SIZE <= SHM_SIZE);
        assert!(MAX_LEDGER_ENTRIES > 0);
        assert!(LAYER_SLOTS_OFFSET >= HEADER_SIZE);
        assert!(WORLD_MODEL_OFFSET + mem::size_of::<WorldModel>() <= SHM_SIZE);
        assert!(THOUGHT_BUFFER_OFFSET + mem::size_of::<ThoughtBuffer>() <= SHM_SIZE);
        assert!(LORE_BUFFER_OFFSET + mem::size_of::<LoreBuffer>() <= SHM_SIZE);
        assert!(ACTION_HISTORY_OFFSET + mem::size_of::<ActionHistory>() <= SHM_SIZE);
        // Thought Theater
        assert!(THEATER_OFFSET + mem::size_of::<TheaterHeader>() <= SHM_SIZE);
        assert!(BIG_GRAPH_OFFSET + mem::size_of::<BigGraph>() <= SHM_SIZE);
        assert!(SMALL_GRAPH_OFFSET + mem::size_of::<SmallGraph>() <= SHM_SIZE);
        assert!(OP_SLICE_OFFSET + mem::size_of::<OperationalSlice>() <= SHM_SIZE);
        assert!(PRESSURE_OFFSET + mem::size_of::<PressureBuffer>() <= SHM_SIZE);
        assert!(TRIM_MASK_OFFSET + mem::size_of::<TrimMask>() <= SHM_SIZE);
    }

    #[test]
    fn layer_slot_size_fits() {
        let total_slots = NUM_LAYERS * LAYER_SLOT_SIZE;
        assert!(
            LAYER_SLOTS_OFFSET + total_slots <= LEDGER_OFFSET,
            "layer slots overlap with ledger"
        );
    }

    #[test]
    fn create_and_open() {
        // Use a unique name to avoid collisions with concurrent tests.
        let name = format!("/qualia_test_{}", std::process::id());

        // Clean up any stale region from a previous failed test.
        {
            let c_name = CString::new(name.as_str()).unwrap();
            unsafe { libc::shm_unlink(c_name.as_ptr()) };
        }

        let region = ShmRegion::create(&name).expect("create failed");
        assert_eq!(region.header().magic, SHM_MAGIC);
        assert_eq!(region.header().version, 1);
        assert_eq!(region.header().num_layers, NUM_LAYERS as u32);

        // Open as a second process would.
        let region2 = ShmRegion::open(&name).expect("open failed");
        assert_eq!(region2.header().magic, SHM_MAGIC);

        drop(region2);
        drop(region); // owner unlinks
    }

    #[test]
    fn double_buffer_semantics() {
        let name = format!("/qualia_dbuf_{}", std::process::id());
        {
            let c_name = CString::new(name.as_str()).unwrap();
            unsafe { libc::shm_unlink(c_name.as_ptr()) };
        }

        let region = ShmRegion::create(&name).expect("create failed");
        let slot = region.layer_slot(0);
        let writer = LayerWriter::new(slot);
        let reader = LayerReader::new(slot);

        // Write to back buffer, publish.
        {
            let buf = writer.back_buffer();
            buf.mean[0] = 42.0;
            buf.layer = 0;
        }
        writer.publish();

        // Reader should now see the published data.
        let front = reader.read();
        assert_eq!(front.mean[0], 42.0);

        drop(region);
    }

    #[test]
    fn ledger_append() {
        let name = format!("/qualia_ledger_{}", std::process::id());
        {
            let c_name = CString::new(name.as_str()).unwrap();
            unsafe { libc::shm_unlink(c_name.as_ptr()) };
        }

        let region = ShmRegion::create(&name).expect("create failed");
        assert_eq!(region.ledger_seq(), 0);

        // SAFETY: LedgerEntry is Copy and repr(C); zeroed is valid.
        let mut entry: LedgerEntry = unsafe { mem::zeroed() };
        entry.seq = 0;
        entry.layer = 1;
        entry.event = LedgerEvent::Challenge;
        entry.vfe = 1.5;

        region.append_ledger(&entry);
        assert_eq!(region.ledger_seq(), 1);

        let read_back = region.ledger_entry(0);
        assert_eq!(read_back.layer, 1);
        assert_eq!(read_back.vfe, 1.5);

        drop(region);
    }
}
