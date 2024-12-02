#include "logger.h"

// stores whether logging is enabled globally
bool Logger::logging_enabled = false;

// initializes the logger with the enabled flag
void Logger::initialize(bool enabled) {
    logging_enabled = enabled;
}

// logs a message if logging is enabled and rank conditions are met
void Logger::log(int rank_to_print, int rank, const std::string& msg) {
    if (logging_enabled && (rank == rank_to_print || rank_to_print == -1)) {
        std::cout << "[Rank " << rank << "] " << msg << std::endl;
    }
}