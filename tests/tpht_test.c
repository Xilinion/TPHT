#include "tpht_test_modules.h"

#include <stdio.h>

int main(void) {
    tpht_test_run_config_module();
    tpht_test_run_constructor_module();
    tpht_test_run_api_edges_module();
    tpht_test_run_deterministic_module();
    tpht_test_run_random_model_module();
    tpht_test_run_saturation_module();
    tpht_test_run_churn_module();
    tpht_test_run_thread_module();

    puts("tpht modular exhaustive tests passed");
    return 0;
}
