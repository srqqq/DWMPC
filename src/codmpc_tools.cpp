#include "controllers/dwmpc/codmpc_tools.hpp"

TimerManager::TimerInfo::TimerInfo() : start(Clock::now()) {}

void TimerManager::start(const std::string& name) {
    timers[name].start = Clock::now();
}

double TimerManager::stop(const std::string& name) {
    auto& t = timers[name];
    auto now = Clock::now();

    t.lastMs = std::chrono::duration<double, std::milli>(now - t.start).count();
    t.totalMs += t.lastMs;

    if (t.lastMs < t.minMs) {
        t.minMs = t.lastMs;
    }
    if (t.lastMs > t.maxMs) {
        t.maxMs = t.lastMs;
    }

    ++t.count;
    return t.lastMs;
}

double TimerManager::lastMs(const std::string& name) const {
    auto it = timers.find(name);
    return (it != timers.end()) ? it->second.lastMs : 0.0;
}

double TimerManager::avgMs(const std::string& name) const {
    auto it = timers.find(name);
    if (it == timers.end() || it->second.count == 0) {
        return 0.0;
    }
    return it->second.totalMs / it->second.count;
}

double TimerManager::minMs(const std::string& name) const {
    auto it = timers.find(name);
    if (it == timers.end() || it->second.count == 0) {
        return 0.0;
    }
    return it->second.minMs;
}

double TimerManager::maxMs(const std::string& name) const {
    auto it = timers.find(name);
    if (it == timers.end() || it->second.count == 0) {
        return 0.0;
    }
    return it->second.maxMs;
}

int TimerManager::count(const std::string& name) const {
    auto it = timers.find(name);
    return (it != timers.end()) ? it->second.count : 0;
}

void TimerManager::print(const std::string& name) const {
    auto it = timers.find(name);
    if (it == timers.end() || it->second.count == 0) {
        std::cout << "[" << name << "] 暂无数据\n";
        return;
    }

    std::cout << "[" << name << "]\n";
    std::cout << "当前时间: " << it->second.lastMs << " ms\n";
    std::cout << "平均时间: " << avgMs(name) << " ms\n";
    std::cout << "最小时间: " << minMs(name) << " ms\n";
    std::cout << "最大时间: " << maxMs(name) << " ms\n";
    std::cout << "记录次数: " << count(name) << "\n";
}