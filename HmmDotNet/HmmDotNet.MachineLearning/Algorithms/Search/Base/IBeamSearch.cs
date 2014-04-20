using System.Collections.Generic;

namespace HmmDotNet.MachineLearning.Algorithms.Search.Base
{    
    public interface IBeamLikeSearch<T, M>
    {
        /// <summary>
        ///     Depth of search. When set to zero than it is unlimited
        /// </summary>
        int Depth { get; }
        /// <summary>
        ///     Number of Nodes to be examined at each stage
        /// </summary>
        int BeamWidth { get; }
        /// <summary>
        ///     Heuristic function that gives numerical value to current node
        /// </summary>
        /// <param name="node"></param>
        /// <param name="model"></param>
        /// <returns></returns>
        double HeuristicFunction(T node, M model);
        /// <summary>
        ///     Select candidates for current level
        /// </summary>
        /// <returns></returns>
        IList<T> SelectCandidatesForNextLevel();
    }
}
