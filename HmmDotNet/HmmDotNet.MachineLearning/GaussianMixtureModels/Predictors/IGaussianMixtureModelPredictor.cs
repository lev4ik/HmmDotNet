using HmmDotNet.MachineLearning.Base;

namespace HmmDotNet.MachineLearning.GaussianMixtureModels.Predictors
{
    public interface IGaussianMixtureModelPredictor
    {
        /// <summary>
        ///     Given trained model and prediction request this method runs prediction algroithms
        ///     and returnes prediction result
        /// </summary>
        /// <param name="model">Trained model for prediction</param>
        /// <param name="request">Prediction request data structure</param>
        /// <returns></returns>
        IPredictionResult Predict(IGaussianMixtureModelState model, IPredictionRequest request);
        /// <summary>
        ///     Evaluated prediction result and returns evaluation result data structure5
        /// </summary>
        /// <typeparam name="T">Type of observation</typeparam>
        /// <param name="request">Evaluation request data structure</param>
        /// <returns></returns>
        IEvaluationResult Evaluate(IEvaluationRequest request);
    }
}
