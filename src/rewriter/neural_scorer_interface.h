// src/rewriter/neural_scorer_interface.h
#ifndef MOZC_REWRITER_NEURAL_SCORER_INTERFACE_H_
#define MOZC_REWRITER_NEURAL_SCORER_INTERFACE_H_

#include <string>
#include <vector>
#include "absl/strings/string_view.h"

namespace mozc {

class NeuralScorerInterface {
 public:
  struct ScoredCandidate {
    std::string value;     // 候補の表記（例: "機械"）
    float neural_score;    // ニューラルスコア（高いほど良い候補）
  };

  virtual ~NeuralScorerInterface() = default;

  // 読み文字列と候補リストを受け取り、各候補のスコアを返す。
  // 候補の順序は入力と同じ。スコアは正規化されていない実数値。
  // 実装は必ずスレッドセーフであること（Rewriterはconstメソッドから呼ばれる）。
  virtual std::vector<ScoredCandidate> Score(
      absl::string_view reading,
      const std::vector<absl::string_view>& candidate_values) const = 0;

  // モデルが利用可能か否か（ロード失敗時の安全弁）
  virtual bool IsAvailable() const = 0;
};

}  // namespace mozc
#endif  // MOZC_REWRITER_NEURAL_SCORER_INTERFACE_H_
