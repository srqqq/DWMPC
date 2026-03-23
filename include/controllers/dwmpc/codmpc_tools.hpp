#ifndef CODMPC_TOOLS_HPP
#define CODMPC_TOOLS_HPP

#include <chrono>
#include <string>
#include <unordered_map>
#include <limits>
#include <iostream>

class TimerManager {
private:
    using Clock = std::chrono::high_resolution_clock;

    struct TimerInfo {
        Clock::time_point start;
        double lastMs = 0.0;
        double totalMs = 0.0;
        double minMs = std::numeric_limits<double>::max();
        double maxMs = 0.0;
        int count = 0;

        TimerInfo();
    };

    std::unordered_map<std::string, TimerInfo> timers;

public:
    void start(const std::string& name);
    double stop(const std::string& name);

    double lastMs(const std::string& name) const;
    double avgMs(const std::string& name) const;
    double minMs(const std::string& name) const;
    double maxMs(const std::string& name) const;
    int count(const std::string& name) const;

    void print(const std::string& name) const;
};

#endif // CODMPC_TOOLS_HPP
