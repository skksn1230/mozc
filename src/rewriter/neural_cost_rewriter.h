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

  int capability(const ConversionRequest& request) const override;

  bool Rewrite(const ConversionRequest& request,
               Segments* segments) const override;

 private:
  bool RewriteSegment(converter::Segment* segment) const;

  std::unique_ptr<NeuralScorerInterface> scorer_;

  static constexpr float kNeuralWeight = 0.25f;
  static constexpr float kScoreToMozcCost = 200.0f;
  static constexpr int kMaxCandidatesToScore = 10;
};

}  // namespace mozc
#endif  // MOZC_REWRITER_NEURAL_COST_REWRITER_H_
