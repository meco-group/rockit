#define CASADI_MAX_NUM_THREADS 1

/////////////////////////////////////////////////////////////
#include "acados_solver_rockit_model.h"

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

#include "rockit_config.h"
#include "stdio.h"

#define DENSE_SPARSITY_FMT(n_rows, n_cols)               \
    static casadi_int fmt[3 + n_cols + n_rows * n_cols]; \
    casadi_int count = 0;                                \
    fmt[count++] = n_rows;                               \
    fmt[count++] = n_cols;                               \
    for (casadi_int i = 0; i < n_cols + 1; i++)          \
    {                                                    \
        fmt[count++] = i * n_rows;                       \
    }                                                    \
    for (casadi_int i = 0; i < n_cols; i++)              \
    {                                                    \
        for (casadi_int j = 0; j < n_rows; j++)          \
        {                                                \
            fmt[count++] = j;                            \
        }                                                \
    }

#ifndef CASADI_SYMBOL_EXPORT
#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#if defined(STATIC_LINKED)
#define CASADI_SYMBOL_EXPORT
#else
#define CASADI_SYMBOL_EXPORT __declspec(dllexport)
#endif
#elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
#define CASADI_SYMBOL_EXPORT __attribute__((visibility("default")))
#else
#define CASADI_SYMBOL_EXPORT
#endif
#endif
#ifdef __cplusplus
extern "C"
{
#endif

    struct acados_driver_memory {
        rockit_model_solver_capsule capsule;

        ocp_nlp_config *nlp_config;
        ocp_nlp_dims *nlp_dims;
        ocp_nlp_in *nlp_in;
        ocp_nlp_out *nlp_out;
        ocp_nlp_solver *nlp_solver;
        void *nlp_opts;
    };

    static int acados_driver_mem_counter = 0;
    static int acados_driver_unused_stack_counter = -1;
    static int acados_driver_unused_stack[CASADI_MAX_NUM_THREADS];
    static struct acados_driver_memory acados_driver_mem[CASADI_MAX_NUM_THREADS];


    enum INPUTS {
        IN_LBX,
        IN_UBX,
        IN_LBU,
        IN_UBU,
        IN_C,
        IN_D,
        IN_LG,
        IN_UG,
        IN_LH,
        IN_UH,
        IN_LBX_E,
        IN_UBX_E,
        IN_C_E,
        IN_LG_E,
        IN_UG_E,
        IN_LH_E,
        IN_UH_E,
        IN_LH_0,
        IN_UH_0,
        IN_LBX_0,
        IN_UBX_0,
        IN_X0,
        IN_U0,
        IN_P_GLOBAL,
        IN_P_LOCAL,
        IN_N
    };
    enum OUTPUTS {
        OUT_X,
        OUT_U,
        OUT_N
    };


    CASADI_SYMBOL_EXPORT int acados_driver_init_mem(int mem);
    CASADI_SYMBOL_EXPORT casadi_int acados_driver(const casadi_real **arg, casadi_real **res, casadi_int *iw, casadi_real *w, int mem) {
        struct acados_driver_memory* m = &acados_driver_mem[mem];
        ocp_nlp_config *nlp_config = m->nlp_config;
        ocp_nlp_dims *nlp_dims = m->nlp_dims;
        ocp_nlp_in *nlp_in = m->nlp_in;
        ocp_nlp_out *nlp_out = m->nlp_out;
        void *nlp_opts = m->nlp_opts;
        
        
        if (arg[IN_LBX]) {
            for (int i=1;i<ROCKIT_N;++i) {
                ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbx", (void*) (arg[IN_LBX]+i*MMAP_LBX_SIZE1));
            }
        }
        if (arg[IN_UBX]) {
            for (int i=1;i<ROCKIT_N;++i) {
                ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubx", (void*) (arg[IN_UBX]+i*MMAP_UBX_SIZE1));
            }
        }

        if (arg[IN_LBX_0]) {
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", (void*) arg[IN_LBX_0]);
        }
        if (arg[IN_UBX_0]) {
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", (void*) arg[IN_UBX_0]);
        }

        if (arg[IN_LBX_E]) {
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, ROCKIT_N, "lbx", (void*) arg[IN_LBX_E]);
        }
        if (arg[IN_UBX_E]) {
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, ROCKIT_N, "ubx", (void*) arg[IN_UBX_E]);
        }
        if (arg[IN_C]) {
            for (int i=0;i<ROCKIT_N;++i) {
                ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "C", (void*) (arg[IN_C]+i*MMAP_C_SIZE1*ROCKIT_X_SIZE1));
            }
        }
        
        if (arg[IN_C_E]) {
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, ROCKIT_N, "C", (void*) arg[IN_C_E]);
        }
        if (arg[IN_D]) {
            for (int i=0;i<ROCKIT_N;++i) {
                ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "D", (void*) (arg[IN_D]+i*MMAP_D_SIZE1*ROCKIT_U_SIZE1));
            }
        }
        if (arg[IN_LG]) {
            for (int i=0;i<ROCKIT_N;++i) {
                ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lg", (void*) (arg[IN_LG]+i*MMAP_LG_SIZE1));
            }
        }
        if (arg[IN_LG_E]) {
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, ROCKIT_N, "lg", (void*) arg[IN_LG_E]);
        }
        if (arg[IN_UG]) {
            for (int i=0;i<ROCKIT_N;++i) {
                ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ug", (void*) (arg[IN_UG]+i*MMAP_UG_SIZE1));
            }
        }
        if (arg[IN_UG_E]) {
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, ROCKIT_N, "ug", (void*) arg[IN_UG_E]);
        }
        if (arg[IN_LH]) {
            for (int i=0;i<ROCKIT_N;++i) {
                ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lh", (void*) (arg[IN_LH]+i*MMAP_LH_SIZE1));
            }
        }
        if (arg[IN_LH_E]) {
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, ROCKIT_N, "lh", (void*) arg[IN_LH_E]);
        }
        if (arg[IN_LH_0]) {
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lh", (void*) arg[IN_LH_0]);
        }
        if (arg[IN_UH]) {
            for (int i=0;i<ROCKIT_N;++i) {
                ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "uh", (void*) (arg[IN_UH]+i*MMAP_UH_SIZE1));
            }
        }
        if (arg[IN_UH_E]) {
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, ROCKIT_N, "uh", (void*) arg[IN_UH_E]);
        }
        if (arg[IN_UH_0]) {
            ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "uh", (void*) arg[IN_UH_0]);
        }

        if (arg[IN_X0]) {
            for (int i=0;i<ROCKIT_N+1;++i) {
                ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", (void*) (arg[IN_X0]+i*ROCKIT_X_SIZE1));
            }
        }
        if (arg[IN_U0]) {
            for (int i=0;i<ROCKIT_N;++i) {
                ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", (void*) (arg[IN_U0]+i*ROCKIT_U_SIZE1));
            }
        }

        double *p = w;
        for (int k=0;k<ROCKIT_P_GLOBAL_SIZE1;++k) p[k] = arg[IN_P_GLOBAL][k];
        for (int i=0;i<ROCKIT_N+1;++i) {
            for (int k=0;k<ROCKIT_P_LOCAL_SIZE1;++k) p[ROCKIT_P_GLOBAL_SIZE1+k] = arg[IN_P_LOCAL][k+i*ROCKIT_P_LOCAL_SIZE1];
            rockit_model_acados_update_params(&m->capsule, i, p, ROCKIT_P_GLOBAL_SIZE1+ROCKIT_P_LOCAL_SIZE1);
        }
        //int rti_phase = 0;

        //ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_phase", &rti_phase);
        rockit_model_acados_solve(&m->capsule);

        //rockit_model_acados_print_stats(&m->capsule);

        if (res[OUT_X]) {
            for (int i=0;i<ROCKIT_N+1;++i) {
                ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, i, "x",res[OUT_X]+i*ROCKIT_X_SIZE1);
            }
        }
        if (res[OUT_U]) {
            for (int i=0;i<ROCKIT_N;++i) {
                ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, i, "u",res[OUT_U]+i*ROCKIT_U_SIZE1);
            }
        }


        return 0;
    }

    CASADI_SYMBOL_EXPORT casadi_int acados_driver_n_in(void) {
        return IN_N;
    }

    CASADI_SYMBOL_EXPORT casadi_int acados_driver_n_out(void) {
        return OUT_N;
    }

    CASADI_SYMBOL_EXPORT const casadi_int *acados_driver_sparsity_in(casadi_int i) {
        switch (i) {
        case IN_LBX:
        {
            DENSE_SPARSITY_FMT(MMAP_LBX_SIZE1, MMAP_LBX_SIZE2)
            return fmt;
        }
        case IN_UBX:
        {
            DENSE_SPARSITY_FMT(MMAP_UBX_SIZE1, MMAP_UBX_SIZE2)
            return fmt;
        }
        case IN_LBU:
        {
            DENSE_SPARSITY_FMT(MMAP_LBU_SIZE1, MMAP_LBU_SIZE2)
            return fmt;
        }
        case IN_UBU:
        {
            DENSE_SPARSITY_FMT(MMAP_UBU_SIZE1, MMAP_UBU_SIZE2)
            return fmt;
        }
        case IN_C:
        {
            DENSE_SPARSITY_FMT(MMAP_C_SIZE1, MMAP_C_SIZE2)
            return fmt;
        }
        case IN_D:
        {
            DENSE_SPARSITY_FMT(MMAP_D_SIZE1, MMAP_D_SIZE2)
            return fmt;
        }
        case IN_LG:
        {
            DENSE_SPARSITY_FMT(MMAP_LG_SIZE1, MMAP_LG_SIZE2)
            return fmt;
        }
        case IN_UG:
        {
            DENSE_SPARSITY_FMT(MMAP_UG_SIZE1, MMAP_UG_SIZE2)
            return fmt;
        }
        case IN_LH:
        {
            DENSE_SPARSITY_FMT(MMAP_LH_SIZE1, MMAP_LH_SIZE2)
            return fmt;
        }
        case IN_UH:
        {
            DENSE_SPARSITY_FMT(MMAP_UH_SIZE1, MMAP_UH_SIZE2)
            return fmt;
        }
        case IN_LBX_E:
        {
            DENSE_SPARSITY_FMT(MMAP_LBX_E_SIZE, 1)
            return fmt;
        }
        case IN_UBX_E:
        {
            DENSE_SPARSITY_FMT(MMAP_UBX_E_SIZE, 1)
            return fmt;
        }
        case IN_C_E:
        {
            DENSE_SPARSITY_FMT(MMAP_C_E_SIZE1, MMAP_C_E_SIZE2)
            return fmt;
        }
        case IN_LG_E:
        {
            DENSE_SPARSITY_FMT(MMAP_LG_E_SIZE, 1)
            return fmt;
        }
        case IN_UG_E:
        {
            DENSE_SPARSITY_FMT(MMAP_UG_E_SIZE, 1)
            return fmt;
        }
        case IN_LH_E:
        {
            DENSE_SPARSITY_FMT(MMAP_LH_E_SIZE, 1)
            return fmt;
        }
        case IN_UH_E:
        {
            DENSE_SPARSITY_FMT(MMAP_UH_E_SIZE, 1)
            return fmt;
        }
        case IN_LH_0:
        {
            DENSE_SPARSITY_FMT(MMAP_LH_0_SIZE, 1)
            return fmt;
        }
        case IN_UH_0:
        {
            DENSE_SPARSITY_FMT(MMAP_UH_0_SIZE, 1)
            return fmt;
        }
        case IN_LBX_0:
        {
            DENSE_SPARSITY_FMT(MMAP_LBX_0_SIZE, 1)
            return fmt;
        }
        case IN_UBX_0:
        {
            DENSE_SPARSITY_FMT(MMAP_UBX_0_SIZE, 1)
            return fmt;
        }
        case IN_X0:
        {
            DENSE_SPARSITY_FMT(ROCKIT_X_SIZE1, (ROCKIT_N+1))
            return fmt;
        }
        case IN_U0:
        {
            DENSE_SPARSITY_FMT(ROCKIT_U_SIZE1, ROCKIT_N)
            return fmt;
        }
        case IN_P_GLOBAL:
        {
            DENSE_SPARSITY_FMT(ROCKIT_P_GLOBAL_SIZE1, ROCKIT_P_GLOBAL_SIZE2)
            return fmt;
        }
        case IN_P_LOCAL:
        {
            DENSE_SPARSITY_FMT(ROCKIT_P_LOCAL_SIZE1, ROCKIT_P_LOCAL_SIZE2)
            return fmt;
        }
        default:
            return 0;
        }
    }

    CASADI_SYMBOL_EXPORT const casadi_int *acados_driver_sparsity_out(casadi_int i) {
        switch (i) {
        case OUT_X:
        {
            DENSE_SPARSITY_FMT(ROCKIT_X_SIZE1, (ROCKIT_N+1))
            return fmt;
        }
        case OUT_U:
        {
            DENSE_SPARSITY_FMT(ROCKIT_U_SIZE1, ROCKIT_N)
            return fmt;
        }
        default:
            return 0;
        }
    }

    CASADI_SYMBOL_EXPORT const char *acados_driver_name_in(casadi_int i) {
        switch (i) {
        case IN_LBX: return "lbx";
        case IN_UBX: return "ubx";
        case IN_LBU: return "lbu";
        case IN_UBU: return "ubu";
        case IN_C: return "C";
        case IN_D: return "D";
        case IN_LG: return "lg";
        case IN_UG: return "ug";
        case IN_LH: return "lh";
        case IN_UH: return "uh";
        case IN_LBX_E: return "lbx_e";
        case IN_UBX_E: return "ubx_e";
        case IN_C_E: return "C_e";
        case IN_LG_E: return "lg_e";
        case IN_UG_E: return "ug_e";
        case IN_LH_E: return "lh_e";
        case IN_UH_E: return "uh_e";
        case IN_LH_0: return "lh_0";
        case IN_UH_0: return "uh_0";
        case IN_LBX_0: return "lbx_0";
        case IN_UBX_0: return "ubx_0";
        case IN_X0: return "x0";
        case IN_U0: return "u0";
        case IN_P_LOCAL: return "p_local";
        case IN_P_GLOBAL: return "p_global";
        default:
            return 0;
        }
    }

    CASADI_SYMBOL_EXPORT const char *acados_driver_name_out(casadi_int i) {
        switch (i) {
        case OUT_X: return "x";
        case OUT_U: return "u";
        default:
            return 0;
        }
    }


    CASADI_SYMBOL_EXPORT int acados_driver_work(casadi_int *sz_arg, casadi_int *sz_res, casadi_int *sz_iw, casadi_int *sz_w) {

        *sz_arg = 0, *sz_res = 0, *sz_iw = 0, *sz_w = 0;

        *sz_w += ROCKIT_P_GLOBAL_SIZE1+ROCKIT_P_LOCAL_SIZE1;

        return 0;
    }

    // Alloc memory
    CASADI_SYMBOL_EXPORT int acados_driver_alloc_mem(void) {
        return acados_driver_mem_counter++;
    }

    // Clear memory
    CASADI_SYMBOL_EXPORT void acados_driver_free_mem(int mem) {
        struct acados_driver_memory* m = &acados_driver_mem[mem];
        rockit_model_acados_free(&m->capsule);
    }

    CASADI_SYMBOL_EXPORT int acados_driver_checkout(void) {
        int mid;
        if (acados_driver_unused_stack_counter >= 0) {
            return acados_driver_unused_stack[acados_driver_unused_stack_counter--];
        } else {
            if (acados_driver_mem_counter == CASADI_MAX_NUM_THREADS) return -1;
            mid = acados_driver_alloc_mem();
            if (mid < 0) return -1;
            if (acados_driver_init_mem(mid)) return -1;
            return mid;
        }

        return acados_driver_unused_stack[acados_driver_unused_stack_counter--];
    }

    CASADI_SYMBOL_EXPORT void acados_driver_release(int mem) {
        acados_driver_unused_stack[++acados_driver_unused_stack_counter] = mem;
    }

    // Initialize memory
    CASADI_SYMBOL_EXPORT int acados_driver_init_mem(int mem) {
        struct acados_driver_memory* m = &acados_driver_mem[mem];
        if (rockit_model_acados_create(&m->capsule)) return 1;

        m->nlp_config = rockit_model_acados_get_nlp_config(&m->capsule);
        m->nlp_dims = rockit_model_acados_get_nlp_dims(&m->capsule);
        m->nlp_in = rockit_model_acados_get_nlp_in(&m->capsule);
        m->nlp_out = rockit_model_acados_get_nlp_out(&m->capsule);
        m->nlp_solver = rockit_model_acados_get_nlp_solver(&m->capsule);
        m->nlp_opts = rockit_model_acados_get_nlp_opts(&m->capsule);

        #include "after_init.h"
        
        return 0;
    }

#ifdef __cplusplus
}
#endif