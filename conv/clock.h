/* ------------
 * This code is provided solely for the personal and private use of
 * students taking the CSC367H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited.
 * All forms of distribution of this code, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2019 Bogdan Simion
 * -------------
 */

#ifndef CLOCK__H
#define CLOCK__H

class Clock {
public:
  Clock() {
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
  }

  void start() { cudaEventRecord(event_start); }

  float stop() {
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time;
    cudaEventElapsedTime(&time, event_start, event_stop);
    return time;
  }

private:
  cudaEvent_t event_start, event_stop;
};

#endif
