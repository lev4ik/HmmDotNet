using System;
using HmmDotNet.MachineLearning.Base;

namespace HmmDotNet.MachineLearning.HiddenMarkovModels.Predictors
{
    public static class HiddenMarkovModelPredictorFactory
    {
        public static IHiddenMarkovModelPredictor GetPredictor(PredictorType type)
        {
            switch (type)
            {
                case PredictorType.HmmLikelihood:
                    return new LikelihoodBasedPredictor();
                case PredictorType.HmmViterbi:
                    return new ViterbiBasedPredictor();
            }
            throw new ApplicationException("Not implemented");
        }
    }
}
