#include "rewriter/neural_cost_rewriter.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "converter/candidate.h"
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

bool NeuralCostRewriter::RewriteSegment(converter::Segment* segment) const {
  const int candidates_size = static_cast<int>(segment->candidates_size());
  if (candidates_size == 0) {
    return false;
  }

  const int score_count = std::min(candidates_size, kMaxCandidatesToScore);

  std::vector<absl::string_view> values;
  values.reserve(score_count);
  for (int i = 0; i < score_count; ++i) {
    values.push_back(segment->candidate(i).value);
  }

  const auto scored = scorer_->Score(segment->key(), values);

  if (static_cast<int>(scored.size()) != score_count) {
    LOG(ERROR) << "NeuralScorer returned unexpected size: " << scored.size()
               << " (expected " << score_count << ")";
    return false;
  }

  // -----------------------------------------------------------------------
  // コスト補正を各候補に書き込む
  //
  //   new_cost = original_cost
  //            + clamp(round(-neural_score * kScoreToMozcCost * kNeuralWeight),
  //                    -300, +300)
  //
  // Mozcのコストは「小さいほど良い」
  // ニューラルスコアは「大きいほど良い」→ 符号を反転して加算
  // -----------------------------------------------------------------------
  std::vector<std::pair<int /*new_cost*/, int /*original_idx*/>> order;
  order.reserve(score_count);

  for (int i = 0; i < score_count; ++i) {
    converter::Candidate* cand = segment->mutable_candidate(i);
    const float delta_f =
        -scored[i].neural_score * kScoreToMozcCost * kNeuralWeight;
    const int delta = std::clamp(static_cast<int>(delta_f), -300, 300);
    cand->cost += delta;
    order.push_back({cand->cost, i});
  }

  // -----------------------------------------------------------------------
  // Segment には sort_candidates_by_cost() が存在しないため、
  // コスト補正後の順位を計算し、move_candidate() で並び替える。
  //
  // move_candidate(old_idx, new_idx) は候補を old_idx から new_idx へ移動する。
  // 先頭から1つずつ「正しい候補」を持ってくる挿入ソート相当の操作で実現する。
  // -----------------------------------------------------------------------
  std::stable_sort(order.begin(), order.end(),
                   [](const auto& a, const auto& b) {
                     return a.first < b.first;  // cost 昇順
                   });

  bool cost_changed = false;
  for (int new_idx = 0; new_idx < score_count; ++new_idx) {
    const int old_idx = order[new_idx].second;
    if (old_idx != new_idx) {
      segment->move_candidate(old_idx, new_idx);
      // move_candidate で候補が挿入されるため、
      // 後続の order エントリの old_idx を補正する
      for (int j = new_idx + 1; j < score_count; ++j) {
        if (order[j].second >= new_idx && order[j].second < old_idx) {
          ++order[j].second;
        }
      }
      cost_changed = true;
    }
  }

  return cost_changed;
}

}  // namespace mozc
