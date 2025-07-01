#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>

// global flag to control whether logging is enabled
inline bool logging_enabled = false;

// simplified macro for logging
#define LOG(rank_to_print, msg) \
    do { \
        if (logging_enabled && rank == rank_to_print) { \
            std::cout << "[Rank " << rank << "] " << msg << std::endl; \
        } \
    } while (0)

#endif