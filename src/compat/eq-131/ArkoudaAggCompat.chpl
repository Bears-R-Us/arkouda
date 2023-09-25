module ArkoudaAggCompat {
  proc yieldTask() {
    chpl_task_yield();
  }
}
