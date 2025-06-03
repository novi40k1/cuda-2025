#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

AlignedVector GeluOMP(const AlignedVector& input) {
    AlignedVector output(input.size());

    #pragma omp parallel for
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }

    return output;
}
