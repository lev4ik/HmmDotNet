using System;
using HmmDotNet.MachineLearning.Algorithms;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.MachineLearning.Estimators;
using HmmDotNet.MachineLearning.HiddenMarkovModels;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning
{
    public class HiddenMarkovModelGaussianDistribution : HiddenMarkovModel<NormalDistribution>, IUnivariatePredictor<double>, IMachineLearningUnivariateModel
    {
        #region Private Members

        private double _likelihood = double.NaN;
        private bool _isTrained = false;

        #endregion Private Members

        #region Constructors

        public HiddenMarkovModelGaussianDistribution(IModelCreationParameters<NormalDistribution> parameters)
            : base(parameters)
        {

        }

        #endregion Constructors

        /// <summary>
        ///     Adding each symbol to the observation sequence 
        ///     we will train the model again and see which one will be with the 
        ///     highest likelihood
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public PredictionResult Predict(double[] observations)
        {
            if (!_isTrained)
            {
                throw new ApplicationException("Model is not trained , can't predict");
            }
            const double tolerance = 0.01d;
            // TODO : Fix prediction
            return null;
            //return new LikelihoodBasedNormalDistributionPredictor().Predict(observations, Clone() as HiddenMarkovModelGaussianDistribution, tolerance);
        }
        /// <summary>
        ///     Adding each symbol to the observation sequence 
        ///     we will train the model again and see which one will be with the 
        ///     highest likelihood
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public PredictionResult Predict(double[] observations, double[] weights)
        {
            if (!_isTrained)
            {
                throw new ApplicationException("Model is not trained , can't predict");
            }
            const double tolerance = 0.01d;
            //return new LikelihoodBasedNormalDistributionPredictor().Predict(observations, Clone() as HiddenMarkovModelGaussianDistribution, tolerance);
            // TODO : Fix prediction
            return null;
        }

        public double Likelihood
        {
            get { return _likelihood; }
        }

        public void Train(double[] observations)
        {
            if (_pi == null || _transitionProbabilityMatrix == null || _emission == null)
            {
                throw new ApplicationException("Initialize the model with initial valuesss");
            }
            var model = HiddenMarkovModelStateFactory.GetState(new ModelCreationParameters<NormalDistribution> { Pi = _pi, TransitionProbabilityMatrix = _transitionProbabilityMatrix, Emissions = _emission });//new HiddenMarkovModelState<NormalDistribution>(_pi, _transitionProbabilityMatrix, _emission);
            model.Normalized = Normalized;
            var alg = new BaumWelchGaussianDistribution(Helper.Convert(observations), model);
            var estimatedParameters = alg.Run(100, 20);
            _pi = estimatedParameters.Pi;
            _transitionProbabilityMatrix = estimatedParameters.TransitionProbabilityMatrix;
            _emission = estimatedParameters.Emission as NormalDistribution[];
            _likelihood = estimatedParameters.Likelihood;
            _isTrained = true;
        }
    }
}
