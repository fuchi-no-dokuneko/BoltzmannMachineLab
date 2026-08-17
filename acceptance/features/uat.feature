@daily @uat @web
Feature: Daily acceptance of Boltzmann Machine Lab
  The daily laptop verifies network setup, both training modes, frozen inference,
  sampling, validation, and editable learning functions through visible controls.

  Background:
    Given I open the web application at path "/"
    Then the web page title contains "Boltzmann Machine Lab"

  Scenario: Configure and reset a deterministic training network
    Then CSS "#epoch" contains text "0"
    And JavaScript expression "BoltzmannLab.state.weights.length === 100 && BoltzmannLab.state.biases.length === 10" returns true
    And exactly 6 elements match CSS "#patterns .pattern"
    When I click CSS "#patterns .pattern:first-child .bit:first-child"
    Then CSS "#patterns .pattern:first-child .bit:first-child" has text "0"
    When I replace CSS "#visibleCount" with "20"
    And I replace CSS "#hiddenCount" with "1"
    And I click CSS "#reset"
    Then CSS "#visibleCount" has value "12"
    And CSS "#hiddenCount" has value "2"
    And exactly 12 elements match CSS "#inferenceBits button"
    When I execute JavaScript:
      """
      document.querySelectorAll('#patterns input[type="checkbox"]').forEach((item) => {
        if (item.checked) item.click();
      });
      """
    And I click CSS "#step"
    Then CSS "#status" contains text "Enable at least one training pattern."
    And CSS "#epoch" contains text "0"
    When I click CSS "#reset"
    Then JavaScript expression "BoltzmannLab.state.patterns.every((pattern) => pattern.enabled)" returns true

  Scenario: Start pause single-step and reset training
    When I click CSS "#start"
    Then CSS "#start" is disabled
    And CSS "#pause" is enabled
    When I wait for 1 seconds
    Then the numeric text in CSS "#epoch" is greater than 0
    When I click CSS "#pause"
    And I remember JavaScript expression "BoltzmannLab.state.epoch" as "paused epoch"
    And I wait for 1 seconds
    Then JavaScript expression "BoltzmannLab.state.epoch" equals remembered value "paused epoch"
    And CSS "#start" is enabled
    When I click CSS "#reset"
    And I click CSS "#step"
    Then CSS "#epoch" contains text "1"
    And CSS "#phase" contains text "CD reconstruction"
    When I click CSS "#reset"
    Then CSS "#epoch" contains text "0"
    And CSS "#phase" contains text "idle"

  Scenario: Run frozen RBM inference and draw a reconstructed sample
    When I click CSS "#step"
    And I remember JavaScript expression "Array.from(BoltzmannLab.state.weights)" as "trained weights"
    And I click CSS "#runInference"
    Then CSS "#inferenceStatus" eventually contains text "Inference complete at epoch 1"
    And CSS "#inferenceMethod" contains text "Input-conditioned RBM reconstruction"
    And CSS "#probabilitySum" contains text "1.000000"
    And CSS "#distributionStates" contains text "64"
    And exactly 4 elements match CSS "#hiddenPosterior .probability-row"
    And exactly 6 elements match CSS "#visibleReconstruction .probability-row"
    And exactly 12 elements match CSS "#distribution .distribution-row"
    And JavaScript expression "Array.from(BoltzmannLab.state.weights)" equals remembered value "trained weights"
    And CSS "#drawInferenceSample" is enabled
    When I click CSS "#drawInferenceSample"
    Then JavaScript expression "document.getElementById('sampledState').textContent !== '-' && document.getElementById('sampledState').textContent.length === 6" returns true
    When I click CSS "#inferenceBits button:first-child"
    Then CSS "#inferenceStatus" eventually contains text "Inference input changed"
    And CSS "#probabilitySum" contains text "0.000000"
    And CSS "#drawInferenceSample" is disabled

  Scenario: Train and infer with the full Boltzmann machine
    When I choose value "bm" in CSS "#model"
    And I replace CSS "#visibleCount" with "5"
    And I replace CSS "#hiddenCount" with "3"
    And I click CSS "#reset"
    And I click CSS "#step"
    Then CSS "#epoch" contains text "1"
    And CSS "#phase" contains text "free negative phase"
    When I replace CSS "#inferenceBurnIn" with "0"
    And I replace CSS "#inferenceSamples" with "100"
    And I click CSS "#runInference"
    Then CSS "#inferenceStatus" eventually contains text "Weights were not changed"
    And CSS "#inferenceMethod" contains text "Input-conditioned full-BM Gibbs reconstruction"
    And CSS "#probabilitySum" contains text "1.000000"
    And CSS "#distributionStates" contains text "32"
    And exactly 3 elements match CSS "#hiddenPosterior .probability-row"
    And exactly 5 elements match CSS "#visibleReconstruction .probability-row"

  Scenario: Reject broken learning code and restore the original functions
    When I replace CSS "#activationCode" with:
      """
      function broken(sum, temperature) { return 0.5; }
      """
    And I click CSS "#applyCode"
    Then CSS "#codeStatus" contains text "Code error: Expected function activation"
    When I replace CSS "#activationCode" with:
      """
      function activation(sum, temperature) {
        return sum / temperature > 0 ? 0.9 : 0.1;
      }
      """
    And I replace CSS "#updateCode" with:
      """
      function update(weight, positive, negative, learningRate) {
        return weight + learningRate * (positive - negative);
      }
      """
    And I click CSS "#applyCode"
    Then CSS "#codeStatus" contains text "Custom functions active."
    When I click CSS "#runInference"
    Then CSS "#inferenceMethod" contains text "with custom activation"
    And CSS "#probabilitySum" contains text "1.000000"
    When I click CSS "#resetCode"
    Then CSS "#codeStatus" contains text "Original functions active."
