#ifndef MOZC_REWRITER_NEURAL_COST_REWRITER_H_
#define MOZC_REWRITER_NEURAL_COST_REWRITER_H_

#include <memory>
#include "converter/segments.h"
#include "request/conversion_request.h"
#include "rewriter/neural_scorer_interface.h"
#include "rewriter/rewriter_interface.h"

namespace mozc {

class NeuralCostRewriter : public RewriterInterface {
 public:
  explicit NeuralCostRewriter(std::unique_ptr<NeuralScorerInterface> scorer);

  // CONVERSION のみ対象（PREDICTION は除外して高速化）
  int capability(const ConversionRequest& request) const override;

  bool Rewrite(const ConversionRequest& request,
               Segments* segments) const override;

 private:
  bool RewriteSegment(Segment* segment) const;

  std::unique_ptr<NeuralScorerInterface> scorer_;

  // ニューラルスコアの影響度（0.0〜1.0）
  // 最初は控えめな値から始め、Mozcの既存コストを壊さないようにする
  static constexpr float kNeuralWeight = 0.25f;

  // ニューラルスコア 1.0 の差 → Mozc コスト単位 200 に対応
  // （Mozcの内部コストは対数確率 × ~500 程度のスケール）
  static constexpr float kScoreToMozcCost = 200.0f;

  // 処理する候補の上限（速度とのトレードオフ）
  static constexpr int kMaxCandidatesToScore = 10;
};

}  // namespace mozc
#endif  // MOZC_REWRITER_NEURAL_COST_REWRITER_H_
