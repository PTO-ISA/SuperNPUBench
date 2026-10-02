# Element-wise atomic histogram

This PTO 0.59 smoke test uses `ew_mscatter_add` to increment eight `S32`
histogram bins from 128 lanes.  The index tile contains logical element
indices (`lane & 7`), not byte offsets, and every value is one.  Repeated
indices make the atomic requirement observable: each bin must receive 16
increments.  The same index tile is then consumed by `ew_mgather_masked` with
an all-one predicate tile, exercising the indexed read path and predicate
carrier after the atomic write.
