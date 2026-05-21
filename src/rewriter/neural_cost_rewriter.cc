#include "rewriter/neural_cost_rewriter.h"

#include <algorithm>
#include <numeric>
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

  // スコアリング対象の候補の value を収集
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

  // ------------------------------------------------------------------
  // Step 1: Mozcのオリジナルコストに delta を加算する
  //
  //   delta = clamp( round(-neural_score * kScoreToMozcCost * kNeuralWeight),
  //                  -300, +300 )
  //
  //   new_cost = original_cost + delta
  //
  // Mozcのコストは「小さいほど良い」、neural_score は「大きいほど良い」
  // → 符号を反転して加算することで、Mozcの既存コスト計算を
  //   ニューラルスコアが増減させる形になる
  // ------------------------------------------------------------------
  bool cost_changed = false;
  for (int i = 0; i < score_count; ++i) {
    const float delta_f =
        -scored[i].neural_score * kScoreToMozcCost * kNeuralWeight;
    const int delta = std::clamp(static_cast<int>(delta_f), -300, 300);
    if (delta != 0) {
      segment->mutable_candidate(i)->cost += delta;
      cost_changed = true;
    }
  }

  if (!cost_changed) {
    return false;
  }

  // ------------------------------------------------------------------
  // Step 2: コスト補正後の上位 score_count 件を cost 昇順に安定ソートする
  //
  // move_candidate(old_idx, new_idx) は
  //   「old_idx の要素を取り出して new_idx に挿入する」動作。
  //
  // 先頭 (new_idx=0) から1件ずつ「あるべき候補」を持ってくる
  // 選択ソート相当の操作で実現する。
  //
  // move_candidate(cur_pos → new_idx) を実行すると、
  //   cur_pos > new_idx のとき: [new_idx, cur_pos) の要素が1つ後ろにずれる
  //   cur_pos < new_idx のとき: (cur_pos, new_idx] の要素が1つ前にずれる
  // この両方のケースを正しく indices に反映する。
  // ------------------------------------------------------------------
  std::vector<int> indices(score_count);
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
    return segment->candidate(a).cost < segment->candidate(b).cost;
  });

  for (int new_idx = 0; new_idx < score_count; ++new_idx) {
    const int cur_pos = indices[new_idx];
    if (cur_pos == new_idx) {
      continue;
    }

    segment->move_candidate(cur_pos, new_idx);

    // move_candidate 後のインデックスずれを補正する
    if (cur_pos > new_idx) {
      // cur_pos から new_idx へ「手前に」移動した場合:
      // [new_idx, cur_pos) にあった要素が1つ後ろにずれる
      for (int j = new_idx + 1; j < score_count; ++j) {
        if (indices[j] >= new_idx && indices[j] < cur_pos) {
          ++indices[j];
        }
      }
    } else {
      // cur_pos から new_idx へ「後ろに」移動した場合:
      // (cur_pos, new_idx] にあった要素が1つ前にずれる
      for (int j = new_idx + 1; j < score_count; ++j) {
        if (indices[j] > cur_pos && indices[j] <= new_idx) {
          --indices[j];
        }
      }
    }
  }

  return true;
}

}  // namespace mozc
