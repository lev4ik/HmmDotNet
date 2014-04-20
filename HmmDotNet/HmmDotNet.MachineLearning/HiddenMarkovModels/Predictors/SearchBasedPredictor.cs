using System;
using HmmDotNet.Extentions.Base;
using HmmDotNet.Extentions.Data;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.HiddenMarkovModels.Predictors
{
    public class SearchBasedPredictor : IHiddenMarkovModelPredictor
    {
        private IChangeRatioFinder _changeRatioFinder;

        #region Constructors

        public SearchBasedPredictor(IChangeRatioFinder changeRatioFinder)
        {
            _changeRatioFinder = changeRatioFinder;
        }

        #endregion Constructors
        
        public IPredictionResult Predict<TDistribution>(IHiddenMarkovModel<TDistribution> model, IPredictionRequest request) where TDistribution : IDistribution
        {
            if (!request.ValidateAlgorithmSpecificParameters())
            {
                return null;
            }

            var n = Convert.ToInt32(request.AlgorithmSpecificParameters["NumberOfSamplePoints"]);
            var k = Convert.ToInt32(request.AlgorithmSpecificParameters["NumberOfWinningPoints"]);

            //var ratios = GetMaximumChangeRatios(request.TrainingSet);

            //var candidates = CreateStartingPool(request.TestSet[0], n, ratios);
/*            do
            {
                //var winningCandidate = CalculateWinningCandidates(k, candidates, model);
                //candidates = CreateCandidatePoolFromArray(winningCandidate, n, ratios);
            } while (Continue(request, candidates));*/

            var result = new PredictionResult() {Predicted = new double[1][]};
            //result.Predicted[0] = GetBestProbabilityCandidate(candidates);

            return result;
        }

        public IEvaluationResult Evaluate(IEvaluationRequest request)
        {
            throw new System.NotImplementedException();
        }
    }
}
