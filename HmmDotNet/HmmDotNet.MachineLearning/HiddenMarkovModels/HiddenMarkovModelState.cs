using System;
using System.Linq;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;
using HmmDotNet.Statistics.Distributions.Univariate;

namespace HmmDotNet.MachineLearning.Estimators
{
    public enum HiddenMarkovModelStateCreationType
    {
        NumberOfStates = 0,
        NumberOfStatesAndDelta,
        NumberOfStatesAndNumberOfComponents,
        NumberOfStatesAndDeltaAndNumberOfComponents,
        NumberOfStatesAndEmissions,
        NumberOfStatesAndDeltaAndEmissions,
        PiAndTpmAndEmissions
    }

    public class HiddenMarkovModel<TDistribution> : IHiddenMarkovModel<TDistribution> where TDistribution : IDistribution
    {
        #region Private Members

        protected int _numberOfComponents;
        protected double[] _pi;
        protected double[][] _transitionProbabilityMatrix;
        protected TDistribution[] _emission;
        protected double[][] _emissionWeight;
        protected ModelType _modelType;

        #endregion Private Members

        #region Constructors

        public HiddenMarkovModel(IModelCreationParameters<TDistribution> parameters)
        {
            switch (GetModelCreationType(parameters))
            {
                case HiddenMarkovModelStateCreationType.NumberOfStates:
                    Create(parameters.NumberOfStates.Value);
                    break;
                case HiddenMarkovModelStateCreationType.NumberOfStatesAndDelta:
                    Create(parameters.NumberOfStates.Value, parameters.Delta.Value);
                    break;
                case HiddenMarkovModelStateCreationType.NumberOfStatesAndDeltaAndEmissions:
                    Create(parameters.NumberOfStates.Value, parameters.Delta.Value, parameters.Emissions);
                    break;
                case HiddenMarkovModelStateCreationType.NumberOfStatesAndEmissions:
                    Create(parameters.NumberOfStates.Value, parameters.Emissions);
                    break;
                case HiddenMarkovModelStateCreationType.PiAndTpmAndEmissions:
                    Create(parameters.Pi, parameters.TransitionProbabilityMatrix, parameters.Emissions);
                    break;
                case HiddenMarkovModelStateCreationType.NumberOfStatesAndNumberOfComponents:
                    CreateMixture(parameters.NumberOfStates.Value, parameters.NumberOfComponents.Value);
                    break;
                case HiddenMarkovModelStateCreationType.NumberOfStatesAndDeltaAndNumberOfComponents:
                    CreateMixture(parameters.NumberOfStates.Value, parameters.NumberOfComponents.Value, parameters.Delta.Value);
                    break;
                default :
                    throw new InvalidOperationException("Combination of parameters passed is not supported by any Create method");
            }            
        }
 
        #endregion Constructors

        #region Properties

        public ModelType Type
        {
            get { return _modelType; }
        }

        public double[] Pi
        {
            get { return _pi; }
        }

        public double[][] TransitionProbabilityMatrix
        {
            get { return _transitionProbabilityMatrix; }
        }

        public TDistribution[] Emission
        {
            get { return _emission; }
        }

        public double[][] EmissionWeights 
        {
            get { return _emissionWeight; }
        }

        public int C
        {
            get { return _numberOfComponents; }
        }

        public int N
        {
            get { return _pi.Length; }
        }

        public int M
        {
            get
            {
                var emission = _emission[0] as DiscreteDistribution;
                if (emission != null)
                {
                    return emission.Symbols.Length;
                }
                return -1;
            }
        }

        public double Likelihood { get; set; }

        public bool Normalized { get; set; }

        #endregion Properties

        #region ICloneable Implementation

        public object Clone()
        {
            return MemberwiseClone();
        }

        #endregion ICloneable Implementation

        #region IEquatable Implementation

        public bool Equals(IHiddenMarkovModel<TDistribution> other)
        {
            if (!(VectorExtentions.EqualsTo(Pi, other.Pi) && TransitionProbabilityMatrix.EqualsTo(other.TransitionProbabilityMatrix) && N == other.N && M == other.M && Likelihood == other.Likelihood))
            {
                return false;
            }
            return !Emission.Where((t, i) => !t.Equals(other.Emission[i])).Any();
        }

        public override bool Equals(Object obj)
        {
            if (obj == null)
                return false;

            var personObj = obj as HiddenMarkovModel<TDistribution>;
            if (personObj == null)
            {
                return false;
            }

            return Equals(personObj);
        }

        public static bool operator == (HiddenMarkovModel<TDistribution> model1, HiddenMarkovModel<TDistribution> model2)
        {
            if ((object)model1 == null || ((object)model2) == null)
                return Equals(model1, model2);

            return model1.Equals(model2);
        }

        public static bool operator != (HiddenMarkovModel<TDistribution> model1, HiddenMarkovModel<TDistribution> model2)
        {
            if ((object)model1 == null || ((object)model2) == null)
                return ! Equals(model1, model2);

            return ! model1.Equals(model2);
        }

        #endregion IEquatable Implementation

        #region Protected Methods

        protected static HiddenMarkovModelStateCreationType GetModelCreationType(IModelCreationParameters<TDistribution> parameters)
        {
            if (parameters.NumberOfStates.HasValue && !parameters.Delta.HasValue && parameters.Emissions == null)
            {
                if (parameters.NumberOfComponents.HasValue)
                {
                    return HiddenMarkovModelStateCreationType.NumberOfStatesAndNumberOfComponents;
                }
                return HiddenMarkovModelStateCreationType.NumberOfStates;
            }
            if (parameters.NumberOfStates.HasValue && parameters.Delta.HasValue && parameters.Emissions == null)
            {
                if (parameters.NumberOfComponents.HasValue)
                {
                    return HiddenMarkovModelStateCreationType.NumberOfStatesAndDeltaAndNumberOfComponents;
                }
                return HiddenMarkovModelStateCreationType.NumberOfStatesAndDelta;
            }
            if (parameters.NumberOfStates.HasValue && parameters.Emissions != null)
            {
                return HiddenMarkovModelStateCreationType.NumberOfStatesAndEmissions;
            }
            if (parameters.NumberOfStates.HasValue && parameters.Delta.HasValue && parameters.Emissions != null)
            {
                return HiddenMarkovModelStateCreationType.NumberOfStatesAndDeltaAndEmissions;
            }
            return HiddenMarkovModelStateCreationType.PiAndTpmAndEmissions;
        }

        #endregion Protected Methods

        #region Private Methods

        /// <summary>
        ///     Constructor for Ergodic model
        /// </summary>
        /// <param name="numberOfStates"></param>
        private void Create(int numberOfStates)
        {
            _pi = new double[numberOfStates];
            _transitionProbabilityMatrix = new double[numberOfStates][];
            for (var i = 0; i < numberOfStates; i++)
            {
                _pi[i] = 1d / numberOfStates;
                _transitionProbabilityMatrix[i] = new double[numberOfStates];
                for (var j = 0; j < numberOfStates; j++)
                {
                    _transitionProbabilityMatrix[i][j] = 1d / numberOfStates;
                }
            }
            _modelType = ModelType.Ergodic;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="numberOfStates"></param>
        /// <param name="numberOfComponents"></param>
        private void CreateMixture(int numberOfStates, int numberOfComponents)
        {
            _numberOfComponents = numberOfComponents;
            Create(numberOfStates);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="numberOfStates"></param>
        /// <param name="numberOfComponents"></param>
        /// <param name="delta"></param>
        private void CreateMixture(int numberOfStates, int numberOfComponents, int delta)
        {
            _numberOfComponents = numberOfComponents;
            Create(numberOfStates, delta);
        }

        /// <summary>
        ///     Constructor for Ergodic model with given emissions
        /// </summary>
        /// <param name="numberOfStates"></param>
        /// <param name="emission"></param>
        private void Create(int numberOfStates, TDistribution[] emission)
        {
            Create(numberOfStates);
            if (emission.Length != numberOfStates)
            {
                throw new ArgumentException("Emissions length is not equal to number of states");
            }
            _emission = emission;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="numberOfStates"></param>
        /// <param name="delta"></param>
        /// <param name="emission"></param>
        private void Create(int numberOfStates, int delta, TDistribution[] emission)
        {
            Create(numberOfStates, delta);
            if (emission.Length != numberOfStates)
            {
                throw new ArgumentException("Emissions length is not equal to number of states");
            }
            _emission = emission;
        }

        /// <summary>
        ///     Constructor for Right-Left model with parameter delta
        /// </summary>
        /// <param name="numberOfStates"></param>
        /// <param name="delta"></param>
        private void Create(int numberOfStates, int delta)
        {
            _pi = new double[numberOfStates];
            _transitionProbabilityMatrix = new double[numberOfStates][];
            if (delta > numberOfStates)
            {
                throw new ArgumentException("Delta must be less than number of states");
            }
            for (var i = 0; i < numberOfStates; i++)
            {
                _pi[i] = (i == 0) ? 1 : 0;

                _transitionProbabilityMatrix[i] = new double[numberOfStates];
                var numberOfActiveStates = 0;
                for (var j = 0; j < numberOfStates; j++)
                {
                    _transitionProbabilityMatrix[i][j] = (j < i || j > i + delta) ? 0 : -1;
                }
                numberOfActiveStates = (int)_transitionProbabilityMatrix[i].Sum() * -1;

                for (var j = 0; j < numberOfStates; j++)
                {
                    if (_transitionProbabilityMatrix[i][j] == -1)
                    {
                        _transitionProbabilityMatrix[i][j] = 1d / numberOfActiveStates;
                    }
                }
            }
            _modelType = ModelType.LeftRight;
        }

        private void Create(double[] pi, double[][] transitionProbabilityMatrix, TDistribution[] emission)
        {
            _modelType = GetModelType(pi, transitionProbabilityMatrix);
            _pi = pi;
            _transitionProbabilityMatrix = transitionProbabilityMatrix;
            _emission = emission;
        }

        private static ModelType GetModelType(double[] pi, double[][] transitionProbabilityMatrix)
        {
            bool LeftRight, Ergodic;

            LeftRight = pi[0] == 1;
            Ergodic = !pi.Any(x => x == 0) && !transitionProbabilityMatrix.Any(x => x.Any(y => y == 0));

            return Ergodic ? ModelType.Ergodic : (LeftRight) ? ModelType.LeftRight : ModelType.Custom;
        }

        #endregion Private Methods
    }
}
