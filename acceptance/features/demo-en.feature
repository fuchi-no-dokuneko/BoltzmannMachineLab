@demo @english @web
Feature: English key-feature demonstration of Boltzmann Machine Lab

  Scenario: Show separate training and inference views
    Given I begin a recorded demo
    And I open the web application at path "/"
    When I narrate in "en-US" for at least 8 seconds:
      """
      Boltzmann Machine Lab separates model training from inference, so we can inspect what the current learned parameters predict without changing those parameters.
      """
    Then CSS "#epoch" contains text "0"
    When I click CSS "#step"
    Then CSS "#epoch" eventually contains text "1"
    When I narrate in "en-US" for at least 6 seconds:
      """
      A single training step advances the epoch and updates the network state shown above.
      """
    And I click CSS "#runInference"
    Then CSS "#inferenceStatus" eventually contains text "Inference complete at epoch 1"
    And CSS "#probabilitySum" contains text "1.000000"
    When I narrate in "en-US" for at least 9 seconds:
      """
      The inference panel now displays hidden posterior probabilities, visible reconstruction probabilities, and a normalized ranking of reconstructed visible states while the weights remain frozen.
      """
    Then at least 1 elements match CSS "#distribution .distribution-row"
    And I finish the recorded demo
