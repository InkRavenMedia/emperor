const LOG_EXPERIMENT_NAME = /^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$/;

export function isValidLogExperimentName(name: string) {
  return LOG_EXPERIMENT_NAME.test(name);
}
