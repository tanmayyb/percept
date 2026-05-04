#pragma once

#include <chrono>

#ifndef LOG_TIMING_DISABLE

  #define CAPTURE_TIME(var) auto var = std::chrono::steady_clock::now();
  
  #define CALCULATE_DURATION(var, start, end) \
    var = std::chrono::duration<double, std::milli>(end - start).count();

  #define ACCUMULATE_DURATION(accumulator, start, end) \
    accumulator += std::chrono::duration<double, std::milli>(end - start).count();

  #define LOG_OPEN_TIMING_PERF(file, path){ FILE *f_init = fopen(path, "w"); if (f_init) fclose(f_init); } \
    file = fopen(path, "a")

  #define LOG_INFO_TIMING_PERF(file, ...) if(file) fprintf(file, __VA_ARGS__)

  #define LOG_FLUSH_TIMING_PERF(file) if(file) fflush(file)

  #define LOG_CLOSE_TIMING_PERF(file) if (file) { fclose(file); file = nullptr;}

#else

  #define CAPTURE_TIME(var)

  #define CALCULATE_DURATION(var, start, end)

  #define ACCUMULATE_DURATION(accumulator, start, end)

  #define LOG_OPEN_TIMING_PERF(file, path) file = nullptr

  #define LOG_INFO_TIMING_PERF(file, ...)

  #define LOG_FLUSH_TIMING_PERF(file)

  #define LOG_CLOSE_TIMING_PERF(file)

#endif


#ifndef LOG_COSTS_DISABLE

  #define LOG_OPEN_AGENT_COSTS(file, path){ FILE *f_init = fopen(path, "w"); if (f_init) fclose(f_init); } \
    file = fopen(path, "a")

  #define LOG_INFO_AGENT_COSTS(file, ...) if(file) fprintf(file, __VA_ARGS__)

  #define LOG_FLUSH_AGENT_COSTS(file) if(file) fflush(file)

  #define LOG_CLOSE_AGENT_COSTS(file) if (file) { fclose(file); file = nullptr;}

#else

  #define LOG_OPEN_AGENT_COSTS(file, path) file = nullptr

  #define LOG_INFO_AGENT_COSTS(file, ...)

  #define LOG_FLUSH_AGENT_COSTS(file)

  #define LOG_CLOSE_AGENT_COSTS(file)

#endif


#ifndef LOG_BEST_AGENT_DISABLE

  #define LOG_OPEN_BEST_AGENT(file, path){ FILE *f_init = fopen(path, "w"); if (f_init) fclose(f_init); } \
    file = fopen(path, "a")

  #define LOG_INFO_BEST_AGENT(file, ...) if(file) fprintf(file, __VA_ARGS__)

  #define LOG_FLUSH_BEST_AGENT(file) if(file) fflush(file)

  #define LOG_CLOSE_BEST_AGENT(file) if (file) { fclose(file); file = nullptr;}

#else

  #define LOG_OPEN_BEST_AGENT(file, path) file = nullptr

  #define LOG_INFO_BEST_AGENT(file, ...)

  #define LOG_FLUSH_BEST_AGENT(file)

  #define LOG_CLOSE_BEST_AGENT(file)

#endif