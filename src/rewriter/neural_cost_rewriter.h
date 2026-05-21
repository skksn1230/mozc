// src/rewriter/neural_cost_rewriter.h
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
  // NeuralScorerInterface の所有権を受け取る
  explicit NeuralCostRewriter(
      std::unique_ptr<NeuralScorerInterface> scorer);

  // capability: CONVERSION のみ対象（予測変換は除外して高速化）
  int capability(const ConversionRequest& request) const override;

  // Rewrite: Segmentsを受け取り、各候補のcostを補正して再ソートする
  bool Rewrite(const ConversionRequest& request,
               Segments* segments) const override;

 private:
  // 1セグメントに対してコスト補正を適用する
  bool RewriteSegment(Segment* segment) const;

  std::unique_ptr<NeuralScorerInterface> scorer_;

  // ニューラルスコアの影響度（0.0〜1.0）
  // 値が大きいほどニューラルの影響が強い
  // 初期値は控えめな 0.25 を推奨
  static constexpr float kNeuralWeight = 0.25f;

  // ニューラルスコア → Mozcコスト単位への変換係数
  // Mozcの内部コストは対数確率 × ~500 程度のスケール
  // スコア1.0の差がコスト200程度に対応するよう設定
  static constexpr float kScoreToMozcCost = 200.0f;

  // 処理する候補の上限（速度とのトレードオフ）
  static constexpr int kMaxCandidatesToScore = 10;
};

}  // namespace mozc
#endif  // MOZC_REWRITER_NEURAL_COST_REWRITER_H_
