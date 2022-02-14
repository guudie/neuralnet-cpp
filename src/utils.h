#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <random>

// returns a random permutation array from 0 to n-1
int* randomPermutation(int n) {
    if(!n)
        return NULL;
    
    // initialize the array
    int* arr = new int[n];
    std::iota(arr, arr + n, 0);

    // shuffle the indices
    auto rng = std::default_random_engine {};
    std::shuffle(arr, arr + n, rng);

    return arr;
}

// shuffles an array
template<class T = int>
void shuffle(T* arr, int n) {
    if(!n)
        return;
    
    // shuffle the array
    auto rng = std::default_random_engine {};
    std::shuffle(arr, arr + n, rng);
}

// fast inverse square root
float invSqrt(float number){
    union {
        float f;
        uint32_t i;
    } conv;

    float x2;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    conv.f = number;
    conv.i = 0x5f3759df - ( conv.i >> 1 );
    conv.f = conv.f * ( threehalfs - ( x2 * conv.f * conv.f ) );
    return conv.f;
}

#endif