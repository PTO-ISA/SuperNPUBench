- Source files: `inserted_slot.data`, `lookup_keys.data`, `lookup_values.data`
- Capacity: 2,550,000
- Occupied slots: 2,040,001
- Load factor: 0.8000
- Lookup keys: 409,600 (all found in table)

### Statistics

| Metric | Value |
|--------|-------|
| Min probe distance | 0 |
| Max probe distance | 358 |
| Mean | 2.01 |
| Median | 0 |

### Percentiles

| Percentile | Probe Distance |
|------------|----------------|
| P50 | 0 |
| P75 | 1 |
| P90 | 5 |
| P95 | 10 |
| P99 | 29 |
| P99.5 | 41 |
| P99.9 | 79 |
| P100 | 358 |

### Histogram

| Range | Count | Percentage |
|-------|-------|------------|
| [0, 1) | 249,288 | 60.86% |
| [1, 2) | 60,875 | 14.86% |
| [2, 3) | 28,064 | 6.85% |
| [3, 4) | 16,505 | 4.03% |
| [4, 5) | 10,680 | 2.61% |
| [5, 6) | 7,546 | 1.84% |
| [6, 7) | 5,618 | 1.37% |
| [7, 8) | 4,257 | 1.04% |
| [8, 10) | 6,191 | 1.51% |
| [10, 16) | 9,613 | 2.35% |
| [16, 32) | 7,560 | 1.85% |
| [32, 64) | 2,660 | 0.65% |
| [64, 128) | 660 | 0.16% |
| [128, 256) | 78 | 0.02% |
| [256, 512) | 5 | 0.00% |

### Coverage (keys found within N probes)

| kMaxProbe | Keys Found | Coverage | Missed |
|-----------|-----------|----------|--------|
| 1 | 249,288 | 60.86% | 160,312 |
| 2 | 310,163 | 75.72% | 99,437 |
| 4 | 354,732 | 86.60% | 54,868 |
| 8 | 382,833 | 93.47% | 26,767 |
| 16 | 398,637 | 97.32% | 10,963 |
| 32 | 406,197 | 99.17% | 3,403 |
| 64 | 408,857 | 99.82% | 743 |
| 128 | 409,517 | 99.98% | 83 |
| 256 | 409,595 | 99.999% | 5 |
| 358 | 409,600 | 100.000% | 0 |
| 512 | 409,600 | 100.000% | 0 |

> **Note**: Minimum kMaxProbe for 100% coverage is **358**.

---

## Simple Dataset (hashtable_lookup_sim)

- Source files: `simple_inserted_slot.data`, `simple_lookup_keys.data`, `simple_lookup_values.data`
- Capacity: 2,048
- Occupied slots: 1,638
- Load factor: 0.7998
- Lookup keys: 1,024 (all found in table)

### Statistics

| Metric | Value |
|--------|-------|
| Min probe distance | 0 |
| Max probe distance | 8 |
| Mean | 0.89 |
| Median | 0 |

### Percentiles

| Percentile | Probe Distance |
|------------|----------------|
| P50 | 0 |
| P75 | 1 |
| P90 | 3 |
| P95 | 5 |
| P99 | 7 |
| P99.5 | 7 |
| P99.9 | 8 |
| P100 | 8 |

### Histogram

| Range | Count | Percentage |
|-------|-------|------------|
| [0, 1) | 654 | 63.87% |
| [1, 2) | 158 | 15.43% |
| [2, 3) | 83 | 8.11% |
| [3, 4) | 42 | 4.10% |
| [4, 5) | 29 | 2.83% |
| [5, 6) | 25 | 2.44% |
| [6, 7) | 18 | 1.76% |
| [7, 8) | 10 | 0.98% |
| [8, 10) | 5 | 0.49% |

### Coverage (keys found within N probes)

| kMaxProbe | Keys Found | Coverage | Missed |
|-----------|-----------|----------|--------|
| 1 | 654 | 63.87% | 370 |
| 2 | 812 | 79.30% | 212 |
| 4 | 937 | 91.50% | 87 |
| 8 | 1,019 | 99.51% | 5 |
| 9 | 1,024 | 100.000% | 0 |

> **Note**: Minimum kMaxProbe for 100% coverage is **9**.
