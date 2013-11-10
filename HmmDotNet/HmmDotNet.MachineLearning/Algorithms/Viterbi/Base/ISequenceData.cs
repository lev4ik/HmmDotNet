using System.Collections.Generic;
using HmmDotNet.MachineLearning.Base;

namespace HmmDotNet.MachineLearning.Algorithms
{
    /// <summary>
    ///     Encapsulates observation sequnce data and states
    /// </summary>
    public interface ISequenceData
    {
        /// <summary>
        /// 
        /// </summary>
        IList<IObservation> Observations { get; set; }
        /// <summary>
        /// 
        /// </summary>
        IList<IState> States { get; set; }
    }
}
