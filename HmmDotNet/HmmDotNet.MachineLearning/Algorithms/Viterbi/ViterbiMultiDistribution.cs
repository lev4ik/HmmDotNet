using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;
using HmmDotNet.Mathematic.Extentions;
using HmmDotNet.Statistics.Distributions;

namespace HmmDotNet.MachineLearning.Algorithms
{
    public class ViterbiMultiDistribution : ViterbiBase
    {
        public ViterbiMultiDistribution(bool logNormalized)
        {
            Normalized = logNormalized;
        }

        public List<IState> Run(IList<IObservation> observations, IList<IState> states,
                                double[] startDistribution, 
                                double[][] transitionProbabilityMatrix,
                                double[][] distributionWeights,
                                IDistribution[][] distributions)
        {
            var N = states.Count;
            var T = observations.Count;
            var K = distributionWeights[0].Length;

            ComputationPath = new double[T][];
            ReproducePath = new int[T][];
            var mpp = new List<IState>(states.Count);

            // Initialize for the first observation
            ComputationPath[0] = new double[N];
            ReproducePath[0] = new int[N];
            mpp.Add(new State());
            for (var i = 0; i < N; i++)
            {
                var sum = (Normalized) ? double.NaN : 0d;
                for (int k = 0; k < K; k++)
                {
                    if (Normalized)
                    {
                        sum = LogExtention.eLnSum(sum, LogExtention.eLnProduct(LogExtention.eLn(distributionWeights[i][k]), LogExtention.eLn(GetProbability(distributions[i][k], observations, 0))));
                    }
                    else
                    {
                        sum += distributionWeights[i][k] * GetProbability(distributions[i][k], observations, 0);
                    }
                }
                if (Normalized)
                {
                    ComputationPath[0][i] = LogExtention.eLnProduct(LogExtention.eLn(startDistribution[i]), sum);
                }
                else
                {
                    ComputationPath[0][i] = startDistribution[i] * sum;
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
                    var sum = (Normalized) ? double.NaN : 0d;
                    for (var k = 0; k < K; k++)
                    {
                        if (Normalized)
                        {
                            sum = LogExtention.eLnSum(sum, LogExtention.eLnProduct(LogExtention.eLn(distributionWeights[j][k]), LogExtention.eLn(GetProbability(distributions[j][k], observations, 0))));
                        }
                        else
                        {
                            sum += distributionWeights[j][k] * GetProbability(distributions[j][k], observations, 0);
                        }
                    }
                    if (Normalized)
                    {
                        ComputationPath[t][j] = LogExtention.eLnProduct(max, sum);
                    }
                    else
                    {
                        ComputationPath[t][j] = max * sum;
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
