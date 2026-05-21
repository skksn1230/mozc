// src/rewriter/neural_cost_rewriter.cc
#include "rewriter/neural_cost_rewriter.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "converter/segments.h"
#include "request/conversion_request.h"
#include "rewriter/rewriter_interface.h"

namespace mozc {

NeuralCostRewriter::NeuralCostRewriter(
    std::unique_ptr<NeuralScorerInterface> scorer)
    : scorer_(std::move(scorer)) {}

int NeuralCostRewriter::capability(
    const ConversionRequest& request) const {
  // CONVERSION のみ対象。
  // PREDICTION（Tab補完）は候補数が多く、ニューラルの恩恵も薄いため除外。
  return RewriterInterface::CONVERSION;
}

bool NeuralCostRewriter::Rewrite(const ConversionRequest& request,
                                  Segments* segments) const {
  if (!scorer_->IsAvailable()) {
    return false;
  }

  bool modified = false;
  for (size_t i = 0; i < segments->conversion_segments_size(); ++i) {
    if (RewriteSegment(segments->mutable_conversion_segment(i))) {
      modified = true;
    }
  }
  return modified;
}

bool NeuralCostRewriter::RewriteSegment(Segment* segment) const {
  const int candidates_size = segment->candidates_size();
  if (candidates_size == 0) {
    return false;
  }

  // スコアリング対象の候補を収集（上位 kMaxCandidatesToScore 件のみ）
  const int score_count = std::min(candidates_size, kMaxCandidatesToScore);
  std::vector<absl::string_view> values;
  values.reserve(score_count);
  for (int i = 0; i < score_count; ++i) {
    values.push_back(segment->candidate(i).value);
  }

  // ニューラルスコアを取得
  const std::vector<NeuralScorerInterface::ScoredCandidate> scored =
      scorer_->Score(segment->key(), values);

  if (scored.size() != values.size()) {
    LOG(ERROR) << "NeuralScorer returned unexpected size";
    return false;
  }

  // コスト補正を適用
  // Mozcの候補コストは「小さいほど良い」。
  // ニューラルスコアは「大きいほど良い」なので、符号を反転して加算する。
  //
  // 補正式:
  //   new_cost = original_cost + round(-neural_score * kScoreToMozcCost * kNeuralWeight)
  //
  // 例: neural_score = 1.5（他の候補より明らかに良い）の場合
  //   delta = -1.5 * 200.0 * 0.25 = -75
  //   コストが75下がり、順位が上がりやすくなる
  bool cost_changed = false;
  for (int i = 0; i < score_count; ++i) {
    Segment::Candidate* cand = segment->mutable_candidate(i);
    const float delta_f =
        -scored[i].neural_score * kScoreToMozcCost * kNeuralWeight;
    const int delta = static_cast<int>(delta_f);
    if (delta != 0) {
      cand->cost += delta;
      cost_changed = true;
    }
  }

  if (!cost_changed) {
    return false;
  }

  // コスト変更後に再ソート
  // Segment::sort_candidates_by_cost() は cost 昇順でソートする
  segment->sort_candidates_by_cost();
  return true;
}

}  // namespace mozc
