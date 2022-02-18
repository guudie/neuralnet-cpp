#ifndef LOG_H
#define LOG_H

#ifndef DEBUGGING
#define DEBUGGING
#endif

// includes
#include <iostream>
#include <chrono>

// namespaces
using namespace std::chrono;

// defines
#define DEBUG(msg) std::cout<<msg

#endif