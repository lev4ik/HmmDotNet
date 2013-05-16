using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class Viterbi : ViterbiBase
    {
        public Viterbi(bool logNormalized)
        {
            Normalized = logNormalized;
        }

        /// <summary>
        ///     Viterbi algorithm calculates most probable path given an observation sequence. 
        ///     Algorithm complexity is O(T * states.Count ^ 2)
        /// </summary>
        /// <param name="observations">Observations sequnce</param>
        /// <param name="states">List of model hidden states</param>
        /// <param name="startDistribution">Starting distribution probability vector</param>
        /// <param name="transitionProbabilityMatrix">Transition probability matrix, as Dictionary that hold for each state it's transition vector for all other states</param>
        /// <param name="distributions">Vector of observation states distributions</param>
        /// <returns>Most Probable Path</returns>
        public List<IState> Run(IList<IObservation> observations, IList<IState> states, 
                                double[] startDistribution, 
                                double[][] transitionProbabilityMatrix, 
                                IList<IDistribution> distributions)
        {
            var N = states.Count;
            var T = observations.Count;
            ComputationPath = new double[T][];
            ReproducePath = new int[T][];
            var mpp = new List<IState>(states.Count);
            
            // Initialize for the first observation
            ComputationPath[0] = new double[N];
            ReproducePath[0] = new int[N];
            mpp.Add(new State());
            for (var i = 0; i < N; i++)
            {
                if (Normalized)
                {
                    ComputationPath[0][i] = LogExtention.eLnProduct(LogExtention.eLn(startDistribution[i]), LogExtention.eLn(GetProbability(distributions[i], observations, 0)));
                }
                else
                {
                    ComputationPath[0][i] = startDistribution[i] * GetProbability(distributions[i], observations, 0);                    
                }
                ReproducePath[0][i] = -1;
            }            
                        
            // Induction
            for (var t = 1; t < T; t++)
            {
                ComputationPath[t] = new double[N];
                ReproducePath[t] = new int[N];
                mpp.Add(new State());                    
                for (var j = 0; j < N; j++)
                {
                    // argmax + max
                    var max = double.NegativeInfinity; //0;
                    for (var i = 0; i < N; i++)
                    {                        
                        var value = (Normalized) ? LogExtention.eLnProduct(ComputationPath[t - 1][i], LogExtention.eLn(transitionProbabilityMatrix[i][j])) :
                                                      ComputationPath[t - 1][i] * transitionProbabilityMatrix[i][j];
                        if (value > max)
                        {
                            max = value;
                            ReproducePath[t][j] = i;
                        }
                    }
                    if (Normalized)
                    {
                        ComputationPath[t][j] = LogExtention.eLnProduct(max, LogExtention.eLn(GetProbability(distributions[j], observations, t)));
                    }
                    else
                    {
                        ComputationPath[t][j] = max * GetProbability(distributions[j], observations, t);   
                    }
                }
            }
            // Calculate results (from first observation)
            mpp[T - 1] = states[GetMaximumDeltaValueStateIndex(ComputationPath[T - 1])];
            for (var i = T - 2; i >= 0; i--)
            {
                mpp[i] = states[ReproducePath[i + 1][mpp[i + 1].Index]];
            }
            return mpp;
        }
    }
}
