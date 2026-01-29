//
// Created by geraldebmer on 14.06.24.
//

#ifndef PINOPT_TIMER_H
#define PINOPT_TIMER_H


#include <chrono>

class Timer {
    int64_t timer_count = 0;
public:
    Timer() = default;

    void tic() {
        start = std::chrono::high_resolution_clock::now();
    }

    int64_t toc() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        timer_count = duration.count();
        return timer_count;
    }

    int64_t elapsed_time()
    {
        return timer_count;
    }

private:
    std::chrono::high_resolution_clock::time_point start;
};

#endif //PINOPT_TIMER_H
