#include "logger.h"

bool Logger::logging_enabled = false;

void Logger::initialize(bool enabled) {
    logging_enabled = enabled;
}

void Logger::log(int rank_to_print, int rank, const std::string& msg) {
    if (logging_enabled && rank == rank_to_print) {
        std::cout << msg << std::endl;
    }
}
