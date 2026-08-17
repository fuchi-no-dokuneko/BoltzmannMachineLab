@demo @cantonese @web
Feature: Cantonese key-feature demonstration of Boltzmann Machine Lab

  Scenario: 展示分開運作的訓練及推理介面
    Given I begin a recorded demo
    And I open the web application at path "/"
    When I narrate in "yue-HK" for at least 8 seconds:
      """
      呢個玻爾茲曼機實驗室將模型訓練同推理分開，等我哋可以睇清楚目前參數嘅預測結果，而唔會喺推理期間改動權重。
      """
    Then CSS "#epoch" contains text "0"
    When I click CSS "#step"
    Then CSS "#epoch" eventually contains text "1"
    When I narrate in "yue-HK" for at least 6 seconds:
      """
      撳一次單步訓練之後，週期數會增加，網絡狀態亦會即時更新。
      """
    And I click CSS "#runInference"
    Then CSS "#inferenceStatus" eventually contains text "Inference complete at epoch 1"
    And CSS "#probabilitySum" contains text "1.000000"
    When I narrate in "yue-HK" for at least 9 seconds:
      """
      推理區而家會顯示隱藏節點後驗機率、可見層重建機率，同埋總和等於一嘅重建狀態分佈，期間權重會保持不變。
      """
    Then at least 1 elements match CSS "#distribution .distribution-row"
    And I finish the recorded demo
