#define CEIL_DIV(A, B)  (((A) + (B) - 1) / (B))
#define CEIL_TO(A, B)  ((((A) + (B) - 1) / (B)) * (B))


#define B  1
#define M  4096
#define K  1536
#define N  24576
#define M_GROUP_SZ 1
#define N_GROUP_SZ 128
#define K_GROUP_SZ 128
#define M_MX  CEIL_DIV(M, M_GROUP_SZ)
#define K_MX  CEIL_DIV(K, K_GROUP_SZ)
#define N_MX  CEIL_DIV(N, N_GROUP_SZ)
#define B_DIM 1
#define M_DIM 4
#define N_DIM 13
#define K_DIM 1 // not support split K

#define VL 64

#define BPC CEIL_DIV(B, B_DIM)
#define MPC CEIL_TO(CEIL_DIV(M, M_DIM), VL)
#define NPC CEIL_TO(CEIL_DIV(N, N_DIM), N_GROUP_SZ)
#define KPC CEIL_DIV(K, K_DIM)

#define MPC_MX CEIL_DIV(MPC, M_GROUP_SZ)
#define NPC_MX CEIL_DIV(NPC, N_GROUP_SZ)
#define KPC_MX CEIL_DIV(KPC, K_GROUP_SZ)


#define N_REMAINS (N % NPC)

#define last_NPC (N_REMAINS==0? NPC : N_REMAINS)

#define M_REMAINS (M % MPC)

#define last_MPC (M_REMAINS==0? MPC : M_REMAINS)

#define A_MTE2_SIZE 1
#define B_MTE2_SIZE 1
#define C_OUT_SIZE 2

#define A_NZ false
#define B_NZ false
#define A_TRANS false
#define B_TRANS false
#define AMX_NZ false //default
#define BMX_NZ false //default
#define AMX_TRANS false //default
#define BMX_TRANS true //default
#define MTE2_DATA_TYPE float8_e4m3_t
#define MTE2_MX_DATA_TYPE float
#define MTE1_DATA_TYPE_A float8_e4m3_t
#define MTE1_DATA_TYPE_B float8_e4m3_t

#define C_DATA_TYPE float16_t
#define ACCUM_DATA_TYPE float

// #define M_CLST 4
// #define N_CLST 4
#define CORE_IDX block_idx //(block_idx / N_CLST * N_DIM + block_idx % N_CLST)

#define BLOCK_ID_B (CORE_IDX % B_DIM)
#define BLOCK_ID_M (CORE_IDX / N_DIM)
#define BLOCK_ID_N (CORE_IDX % N_DIM)


#define is_last_N (BLOCK_ID_N == (N_DIM - 1))
#define NPC_TO_USE (is_last_N? last_NPC : NPC)


#define is_last_M (BLOCK_ID_M == (M_DIM - 1))
#define MPC_TO_USE (is_last_M? last_MPC : MPC)

// tiling
#define TL1M_FL  128
#define TL1K  512
#define TL1N_FL  128

#define TL1M_RM  (MPC_TO_USE % TL1M_FL)  
#define LOOP_M CEIL_DIV(MPC_TO_USE, TL1M_FL) 
#define TL1M  ((TL1M_RM!=0 & tm == (LOOP_M-1)) ? TL1M_RM : TL1M_FL)

#define TL1N_RM  (NPC_TO_USE % TL1N_FL)
#define LOOP_N   CEIL_DIV(NPC_TO_USE, TL1N_FL) 
#define TL1N  ((TL1N_RM!=0 & tn == (LOOP_N-1)) ? TL1N_RM : TL1N_FL)


#define TL1M_FL_SCREW  (TL1M_FL + (32/C_OUT_SIZE))
#define TL1N_FL_SCREW  (TL1N_FL + (32/C_OUT_SIZE))

#define TL0M  TL1M
#define TL0K  128
#define TL0N  TL1N

#define TL1M_UR16 CEIL_TO(TL1M, 16) //up round to 16
#define TL0M_UR16 CEIL_TO(TL0M, 16)  

#define TL1N_UR16 CEIL_TO(TL1N, 16) //up round to 16
#define TL0N_UR16 CEIL_TO(TL0N, 16)  
//mx buffer tiling
#define TL1M_MX  CEIL_DIV(TL1M , M_GROUP_SZ)
#define TL1K_MX  CEIL_DIV(TL1K , K_GROUP_SZ)
#define TL1N_MX  CEIL_DIV(TL1N , N_GROUP_SZ)

#define TL1M_FL_MX CEIL_DIV(TL1M_FL , M_GROUP_SZ) 
#define TL1N_FL_MX CEIL_DIV(TL1N_FL , N_GROUP_SZ) 

#define TL0M_MX  CEIL_DIV(TL0M , M_GROUP_SZ)
#define TL0K_MX  CEIL_DIV(TL0K , K_GROUP_SZ) 
#define TL0N_MX  CEIL_DIV(TL0N , N_GROUP_SZ)

// Loop
#define LOOP_K CEIL_DIV(KPC, TL1K)

#define LOOP_K_INNER CEIL_DIV(TL1K, TL0K)
#define LOOP_BATCH BPC

// GEMV
#define GEMV_DISABLE 1
// #define L1A_STRIDE (GEMV_DISABLE ? TL1M_UR16 : TL1M)
// #define L0A_STRIDE (GEMV_DISABLE ? TL0M_UR16 : TL0M)


#define UB_ALIGNMENT 32


// #define AIV_TEST true

// #define UNIT_FLAG true

// #define AIV_TEST true
// #define AIC_TEST true



// other params.
#define C0_BYTE (32)
#define FIXP true 
#ifdef AIC_TEST
    #define FIXP_TYPE F322BF16 //VQF322BF16_PRE  
#else
    #define FIXP_TYPE NoQuant //VQF322BF16_PRE  
#endif
#define DEQ_PARAMS_PER_CHN (8)
#define MTE_PRE_ALLOCATE 1 //1
#define VEC_PRE_ALLOCATE 0




#define VSSTB_STRIDE  (((18) << 16) + 144)   // (((block stride) << 16) + ( repeat stride )) in 32B 

// Quant, not consider yet.
// sync, only support unitflag.
// Dataflow, only support os.