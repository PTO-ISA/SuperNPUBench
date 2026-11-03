#pragma once

// Public entry point: four-slot-per-round masked TileOP probing over a table
// with a 48-byte guard tail (see hashtable_lookup_tileop_stride4_guard.hpp).
// Callers must provide three readable entries after slot Capacity-1 that
// mirror slots 0..2 and keep Capacity >= 4.  The unguarded implementation
// remains available in hashtable_lookup_tileop_masked.hpp.
#include "hashtable_lookup_tileop_stride4_guard.hpp"
