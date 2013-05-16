using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;

namespace HmmDotNet.MachineLearning
{
    public class Observation : IObservation
    {
        #region Properties

        public string Description { get; protected set; }

        public double[] Value { get; protected set; }
        
        public int Dimention
        {
            get
            {
                var d = -1;
                if (Value != null)
                {
                    d = Value.Length;
                }
                return d; 
            }
        }

        #endregion Properties

        #region Constructors

        public Observation(double[] value, string description)
        {
            Description = description;
            Value = value;
        }

        #endregion Constructors        
    }

    public static class ObservationUtils
    {
        public static double[] ToUnivariateArray(this IList<IObservation> list)
        {
            var observations = new double[list.Count];
            for (var i = 0; i < list.Count; i++)
            {
                observations[i] = list[i].Value[0];
            }

            return observations;
        }

        public static double[][] ToMultivariateArray(this IList<IObservation> list)
        {
            var observations = new double[list.Count][];
            for (var i = 0; i < list.Count; i++)
            {
                observations[i] = list[i].Value;
            }

            return observations;
        }
    }
}
