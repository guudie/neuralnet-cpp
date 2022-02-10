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

#endif