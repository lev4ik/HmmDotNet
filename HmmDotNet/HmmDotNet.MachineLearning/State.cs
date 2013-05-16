using HmmDotNet.MachineLearning.Base;

namespace HmmDotNet.MachineLearning
{
    public class State : IState
    {
        #region Properties

        public string Description { get; protected set; }
        public int Index { get; protected set; }

        #endregion Properties

        #region Constructors

        public State(int index, string description)
        {
            Description = description;
            Index = index;
        }

        public State()
        {

        }

        #endregion Constructors
    }
}
