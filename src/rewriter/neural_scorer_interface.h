#ifndef MOZC_REWRITER_NEURAL_SCORER_INTERFACE_H_
#define MOZC_REWRITER_NEURAL_SCORER_INTERFACE_H_

#include <string>
#include <vector>
#include "absl/strings/string_view.h"

namespace mozc {

class NeuralScorerInterface {
 public:
  struct ScoredCandidate {
    std::string value;
    float neural_score;  // 高いほど良い候補
  };

  virtual ~NeuralScorerInterface() = default;

  // スレッドセーフであること（Rewrite() は const から呼ばれる）
  virtual std::vector<ScoredCandidate> Score(
      absl::string_view reading,
      const std::vector<absl::string_view>& candidate_values) const = 0;

  virtual bool IsAvailable() const = 0;
};

}  // namespace mozc
#endif  // MOZC_REWRITER_NEURAL_SCORER_INTERFACE_H_
