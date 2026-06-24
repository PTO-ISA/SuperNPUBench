#define B 1
#define M 96
#define K 4096
#define N 28672
#define M_MX  M
#define K_MX  (K/32)
#define N_MX  N
#define B_DIM 1
#define M_DIM 1
#define N_DIM 53
#define K_DIM 1 // not support split K

#define BPC ((B+B_DIM-1)/B_DIM)
#define MPC (((M/M_DIM)+15)/16*16)
#define NPC (((N/N_DIM)+15)/16*16)
#define KPC (K/K_DIM)

#define A_MTE2_SIZE 2
#define B_MTE2_SIZE 1
#define C_OUT_SIZE 2
#define A_SIZE 2
#define B_SIZE 1

#define A_NZ false
#define B_NZ false
#define A_TRANS false
#define B_TRANS true
#define MTE2_DATA_TYPE_A int16_t
#define MTE2_DATA_TYPE_B int8_t
#define MTE1_DATA_TYPE_A bfloat16_t
#define MTE1_DATA_TYPE_B float8_e4m3_t

#define M_CLST 4
#define N_CLST 4
#define CORE_IDX block_idx //(block_idx / N_CLST * N_DIM + block_idx % N_CLST)

#define BLOCK_ID_B (CORE_IDX % B_DIM)
#define BLOCK_ID_M (CORE_IDX / N_DIM)
#define BLOCK_ID_N (CORE_IDX % N_DIM)

// tiling
#define TL1M 96
#define TL1K 512
#define TL1N_FL 272
#define TL1N_RM  (NPC % TL1N_FL)
#define LOOP_N ((NPC + TL1N_FL -1) / TL1N_FL)
#define TL1N  ((TL1N_RM!=0 & tn == LOOP_N-1) ? TL1N_RM : TL1N_FL)

#define TL0M  TL1M
#define TL0K (TL1K/8)
#define TL0N  TL1N

#define TL1M_UR16 ((TL1M+15)/16*16) //up round to 16
#define TL0M_UR16 ((TL0M+15)/16*16)

// Loop
#define LOOP_M (MPC / TL1M) 
#define LOOP_K (KPC / TL1K)

#define LOOP_K_INNER (TL1K/TL0K)
#define LOOP_BATCH BPC

// GEMV
#define GEMV_DISABLE 1
#define L1A_STRIDE (GEMV_DISABLE ? TL1M_UR16 : TL1M)
#define L0A_STRIDE (GEMV_DISABLE ? TL0M_UR16 : TL0M)

// other params.
#define C0_BYTE (32)
#define FIXP true 
#define FIXP_TYPE VQF322BF16_PRE 
#define DEQ_PARAMS_PER_CHN (8)
#define MTE_PRE_ALLOCATE 1

// Quant, not consider yet.
// sync, only support unitflag.
// Dataflow, only support os.