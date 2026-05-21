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

  // 上位 kMaxCandidatesToScore 件のみスコアリング
  const int score_count = std::min(candidates_size, kMaxCandidatesToScore);
  std::vector<absl::string_view> values;
  values.reserve(score_count);
  for (int i = 0; i < score_count; ++i) {
    values.push_back(segment->candidate(i).value);
  }

  const auto scored = scorer_->Score(segment->key(), values);

  if (static_cast<int>(scored.size()) != score_count) {
    LOG(ERROR) << "NeuralScorer returned unexpected size";
    return false;
  }

  // コスト補正を適用
  //
  //   new_cost = original_cost + round(-neural_score * kScoreToMozcCost * kNeuralWeight)
  //
  // Mozcのコストは「小さいほど良い」
  // ニューラルスコアは「大きいほど良い」→ 符号を反転して加算
  //
  // 発散防止: delta を [-300, +300] にクリップ
  bool cost_changed = false;
  for (int i = 0; i < score_count; ++i) {
    Segment::Candidate* cand = segment->mutable_candidate(i);
    const float delta_f =
        -scored[i].neural_score * kScoreToMozcCost * kNeuralWeight;
    const int delta = std::clamp(static_cast<int>(delta_f), -300, 300);
    if (delta != 0) {
      cand->cost += delta;
      cost_changed = true;
    }
  }

  if (!cost_changed) {
    return false;
  }

  // コスト変更後に再ソート（cost 昇順）
  segment->sort_candidates_by_cost();
  return true;
}

}  // namespace mozc
