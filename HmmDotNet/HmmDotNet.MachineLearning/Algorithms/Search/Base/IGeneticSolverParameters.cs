namespace HmmDotNet.MachineLearning.Algorithms.Search.Base
{
    /// <summary>
    ///     Parameters requeired for Genetic Solver initialization
    /// </summary>
    public interface IGeneticSolverParameters
    {
        /// <summary>
        ///     Number of generations that the algorithm will create
        /// </summary>
        int NumberOfGenerations { get; set; }
        /// <summary>
        ///     The size of the population of each generation
        /// </summary>
        int PopulationSize { get; set; }
        /// <summary>
        ///     The size of the tournament for tournament selection procedure
        /// </summary>
        int TournamentSize { get; set; }
        /// <summary>
        ///     Probability of crossover to happen
        /// </summary>
        decimal CrossOverProbability { get; set; }
        /// <summary>
        ///     Probability of mulation to happen
        /// </summary>
        decimal MutationProbability { get; set; }
        /// <summary>
        ///     How many chromosomes will be transfered as is to next level
        /// </summary>
        int BestChromosomeTransferPercent { get; set; }
        /// <summary>
        ///     Number of chromosomes in random population
        /// </summary>
        int RandomPopulationSize { get; set; }
    }
}
