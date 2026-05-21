#ifndef MOZC_REWRITER_STUB_NEURAL_SCORER_H_
#define MOZC_REWRITER_STUB_NEURAL_SCORER_H_

#include <string>
#include <vector>
#include "rewriter/neural_scorer_interface.h"

namespace mozc {

class StubNeuralScorer : public NeuralScorerInterface {
 public:
  bool IsAvailable() const override { return true; }

  std::vector<ScoredCandidate> Score(
      absl::string_view /*reading*/,
      const std::vector<absl::string_view>& candidates) const override {
    std::vector<ScoredCandidate> results;
    results.reserve(candidates.size());
    for (const auto& v : candidates) {
      results.push_back({std::string(v), 0.0f});  // 全候補スコア0 = 変化なし
    }
    return results;
  }
};

}  // namespace mozc
#endif  // MOZC_REWRITER_STUB_NEURAL_SCORER_H_
