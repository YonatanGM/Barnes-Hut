#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <string>

class Logger {
public:
    static void initialize(bool enabled);
    static void log(int rank_to_print, int rank, const std::string& msg);
private:
    static bool logging_enabled;
};

#endif
