module ArkoudaAggCompat {
  proc yieldTask() {
    currentTask.yieldExecution();
  }
}