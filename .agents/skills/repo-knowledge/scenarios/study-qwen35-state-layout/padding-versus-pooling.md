# Padding versus pooling: independent lanes can be smaller here

2026-09-25 source-derived arithmetic for35B TP2/MTP2, per rank. Not measured
HBM savings. Uses the11-slot,3x10GDN+11FA grouping established in GUIDE.md.

Let C=40960bytes conv, S=1048576bytes recurrent or one FA K/V logical block,
P=C+2S. One global block ID reserves a column across11 raw allocations:
11P=22.4296875MiB. It belongs to one live cache group at a time.

- FA group:11*(K+V)=22MiB useful payload;11C=0.4296875MiB padding,
  1.9157% of reserved extent.
- One10-layer GDN group:10*(C+S)=10.390625MiB whole-row payload;
  12.0390625MiB unused,53.6747% of reserved extent. This includes the spare
  V-sized slab per real layer and the group's absent11th layer slot.

Sharing the global free-ID pool allows free capacity to serve either group;
it does NOT let another group use the unused portion of an already owned ID.
Thus heterogeneous pooling reduces stranded *free IDs* but leaves significant
internal padding for GDN-owned IDs. These are different efficiency effects.

Illustrative steady running envelope (not complete request/cache accounting):
three GDN groups x (one running + two speculative rows) x22.4296875MiB =
201.8671875MiB/rank per request in the native pooled geometry. Separate GDN
lanes preserving one extended conv and three recurrent candidates per layer
need30*(C+3S)=91.171875MiB/rank per resident seat. Extra retained checkpoints,
transition rows, inactive scratch, alignment and physical allocator effects are
excluded. Do not claim the difference is measured reclaimable memory in the
current service or multiply by a guessed live request count as a total saving.

SIMD lanes alone do not guarantee lower waste. Equalizing all lanes to a common
logical capacity can strand capacity when token history and seat pressure differ.
The proposed design benefits from BOTH exclusive exact-size lanes AND distinct
seat versus token-page domains. Shared FA/draft page capacity stays bounded;
seat-indexed GDN/candidate State grows with resident count, not token length.
No max-context array is preallocated per seat. No separate draft GDN is assumed:
MTP candidate GDN here belongs to the target's verification, while draft is FA.

The cost of abandoning heterogeneous pooling is reduced fungibility between
those separate budgets. Fixed seat capacity can be idle; token pages can be full.
That tradeoff must be accounted rather than promising every saved padding byte
becomes usable FA capacity. Backend admission and a real physical receipt decide
net savings after the State/graph initialization handoff.
