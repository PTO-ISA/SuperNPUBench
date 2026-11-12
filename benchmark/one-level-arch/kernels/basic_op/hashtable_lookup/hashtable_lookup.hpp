#pragma once

// Public entry point: adaptive two-plus-six masked TileOP probing over a table
// with a 112-byte guard tail (see hashtable_lookup_tileop_adaptive.hpp).
//
// Each round gathers slots o and o+1 for every still-active lane, then gathers
// o+2 .. o+7 only for the lanes that missed both A slots.  The unguarded
// masked implementation remains available in
// hashtable_lookup_tileop_masked.hpp and can be selected with
// -DHASH_KERNEL_HEADER='"..."' when the caller cannot provide a guard tail.
//
// Guard contract: Capacity-1 must be followed by 7 readable 16-byte entries
// mirroring slots 0..6 and Capacity must be >= 4.  With
// -DHASH_ADAPTIVE_TAIL=2 (two-plus-two) the requirement drops back to 3
// entries / 48 bytes.
#include "hashtable_lookup_tileop_adaptive.hpp"
